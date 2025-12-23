# app.py
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import pandas as pd
import os, json, hashlib, random
from io import BytesIO
from datetime import datetime
import imghdr

# Optional PDF support
try:
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ----------------- Config & paths -----------------
st.set_page_config(page_title="AI-Powered PCOD Dashboard", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FILE = os.path.join(BASE_DIR, "users.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "ultrasounds")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure users.json exists with structure {"users": []}
if not os.path.exists(USER_FILE):
    try:
        with open(USER_FILE, "w") as f:
            json.dump({"users": []}, f, indent=2)
    except Exception as e:
        st.error(f"Cannot create users.json. Move folder to a writable location. ({e})")
        st.stop()

# ----------------- User DB helpers -----------------
def load_db():
    try:
        with open(USER_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"users": []}

def save_db(db):
    try:
        with open(USER_FILE, "w") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save users.json: {e}")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def user_exists(username: str) -> bool:
    db = load_db()
    return any(u["username"] == username for u in db.get("users", []))

def add_user(username: str, password: str, extra: dict = None):
    db = load_db()
    if extra is None:
        extra = {}
    db_user = {
        "username": username,
        "password": hash_password(password),
        "created": datetime.utcnow().isoformat(),
        "profile": extra,
        "history": []
    }
    db["users"].append(db_user)
    save_db(db)

def verify_user(username: str, password: str) -> bool:
    db = load_db()
    for u in db.get("users", []):
        if u["username"] == username and u["password"] == hash_password(password):
            return True
    return False

def append_user_history(username: str, record: dict):
    db = load_db()
    for u in db.get("users", []):
        if u["username"] == username:
            u.setdefault("history", []).append(record)
            save_db(db)
            return

# ----------------- Ultrasound validator (relaxed & robust) -----------------
def is_ultrasound_image(pil_img: Image.Image) -> bool:
    """
    Relaxed ultrasound-like validator:
    - Accepts grayscale or grayscale-dominant color (handles tinted exports).
    - Requires some dark-region presence, but relaxed thresholds to accept typical monitor screenshots.
    - Requires reasonable texture (not flat) and minimal resolution.
    """
    try:
        small = pil_img.copy()
        small.thumbnail((512,512))
        arr_rgb = np.array(small.convert("RGB")).astype(np.int16)
    except Exception:
        return False

    # 1) grayscale-dominant check: low inter-channel differences for many pixels
    diff = np.abs(arr_rgb[:,:,0] - arr_rgb[:,:,1]) + np.abs(arr_rgb[:,:,1] - arr_rgb[:,:,2])
    gray_like_fraction = float(np.mean(diff < 30))  # fraction of pixels that are nearly grayscale
    if gray_like_fraction < 0.45:
        # Not strongly grayscale-like, but we still allow some less-gray if other checks pass.
        # Return False early only if very color-dominant.
        color_dom = float(np.mean(diff > 60))
        if color_dom > 0.35:
            return False

    # 2) convert to grayscale and analyze intensity distribution
    gray = small.convert("L")
    g = np.array(gray).astype(np.uint8)

    # relaxed dark-ratio: allow monitor screenshots with UI overlays
    dark_ratio = float(np.mean(g < 150))
    if dark_ratio < 0.12:  # lowered to accept lighter scans with overlays
        return False

    # 3) texture variation: ultrasound has structure; avoid completely flat images
    # ignore a small border area to avoid black edges influencing std
    try:
        h, w = g.shape
        crop = g[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
        texture_std = float(np.std(crop))
    except Exception:
        texture_std = float(np.std(g))
    if texture_std < 7.0:
        return False

    # 4) resolution sanity check
    H, W = pil_img.size[1], pil_img.size[0]  # PIL size is (width, height)
    if H < 160 or W < 160:
        return False

    return True

# ----------------- Ultrasound analyzer (simulated) -----------------
def analyze_ultrasound_image(pil_img: Image.Image):
    # convert to grayscale, basic contrast and median filter
    img = pil_img.convert("L")
    img.thumbnail((512,512))
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    arr = np.array(img).astype(float)
    if arr.max() - arr.min() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    mean = arr.mean(); std = arr.std()
    thresh = mean - 0.15 * std
    binary = (arr < thresh).astype(np.uint8)

    H,W = binary.shape
    labels = np.zeros_like(binary, dtype=int)
    cur = 1
    for y in range(H):
        for x in range(W):
            if binary[y,x] == 1 and labels[y,x] == 0:
                stack = [(y,x)]
                labels[y,x] = cur
                while stack:
                    yy,xx = stack.pop()
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = yy+dy, xx+dx
                        if 0<=ny<H and 0<=nx<W and binary[ny,nx]==1 and labels[ny,nx]==0:
                            labels[ny,nx] = cur
                            stack.append((ny,nx))
                cur += 1
    areas = [int((labels==lbl).sum()) for lbl in range(1,cur)]
    img_area = H*W
    cysts = 0
    for a in areas:
        if a >= max(3, 0.0008*img_area) and a <= max(10, 0.06*img_area):
            cysts += 1

    dark_frac = float(binary.mean())
    score = (dark_frac * 120.0) + min(40, cysts*8) + min(20, std*40)
    score = float(max(0.0, min(100.0, score)))
    debug = Image.fromarray((255*(1-arr)).astype(np.uint8)).convert("RGB")
    return int(cysts), round(score,1), debug

# ----------------- Prognosis logic (cyst-based escalation) -----------------
def compute_prognosis(data: dict, us_score: float):
    bmi = round(data["weight"] / ((data["height"]/100)**2),1) if data.get("height",0)>0 else 0.0
    score = 0.0

    # BMI contribution
    if bmi < 18.5:
        score += 0; bmi_note = "Underweight/low"
    elif bmi < 25:
        score += 5; bmi_note = "Normal BMI"
    elif bmi < 30:
        score += 12; bmi_note = "Overweight"
    else:
        score += 20; bmi_note = "High BMI"

    # Age contribution
    if data.get("age",0) < 20:
        score += 2; age_note = "Younger age"
    elif data.get("age",0) <= 35:
        score += 5; age_note = "Reproductive age"
    else:
        score += 6; age_note = "Older reproductive age"

    # Symptoms
    if data.get("cycle_irregular"):
        score += 18; cycle_note = "Irregular cycles"
    else:
        cycle_note = "Regular cycles"

    if data.get("acne"):
        score += 6; acne_note = "Acne present"
    else:
        acne_note = "No acne"

    if data.get("hair"):
        score += 8; hair_note = "Excess hair/hair fall"
    else:
        hair_note = "No excess hair"

    if data.get("stress_high"):
        score += 4

    if data.get("ovary_volume",0) > 12:
        score += 6

    # Ultrasound contribution scaled
    score += (us_score/100.0) * 20.0
    score = min(max(score, 0.0), 99.9)

    # Base level by numeric score
    if score < 30:
        base_level = "Level A — Normal"; base_color = "#16a34a"
    elif score < 50:
        base_level = "Level B — Moderate"; base_color = "#f59e0b"
    else:
        base_level = "Level C — Severe"; base_color = "#dc2626"

    cysts = int(data.get("cysts", 0))

    # Escalate based on cysts (milder thresholds)
    if cysts >= 8:
        level = "Level C — Severe"; color = "#dc2626"
    elif 3 <= cysts <= 7:
        if "Normal" in base_level:
            level = "Level B — Moderate"; color = "#f59e0b"
        else:
            level = base_level; color = base_color
    else:
        level = base_level; color = base_color

    confidence = round(min(99.9, max(60.0, score + random.uniform(-6,6))),1)
    explanation = {
        "bmi_note": bmi_note, "age_note": age_note, "cycle_note": cycle_note,
        "acne_note": acne_note, "hair_note": hair_note,
        "ovary_note": f"Ovary volume: {data.get('ovary_volume','N/A')} ml",
        "us_score": us_score, "raw_score": round(score,1), "cysts": cysts
    }

    return {
        "level": level,
        "color": color,
        "score": round(score,1),
        "confidence": confidence,
        "explanation": explanation,
        "bmi": bmi
    }


# ----------------- LUNA offline replies -----------------
def luna_reply(msg: str, context: dict):
    t = msg.lower()
    if "egg" in t or "eggs" in t:
        return "Eggs are protein-rich and OK in moderation — prefer boiled/poached."
    if "exercise" in t or "workout" in t:
        return "Aim for 30–45 min daily: walking + strength + yoga."
    if "sleep" in t:
        return "7–8 hours nightly helps hormonal balance."
    if "period" in t or "irregular" in t:
        return f"Irregular cycles often improve with lifestyle changes. Your last result: {context.get('level','N/A')}."
    if "diet" in t:
        return "Low GI carbs, protein each meal, more fiber, reduce refined sugar."
    return random.choice([
        "I can suggest meal swaps, exercises, or explain the report. Ask anything specific!",
        "Tell me about symptoms and I can tailor suggestions."
    ])

# ----------------- Report generation -----------------
def build_text_report(user: str, data: dict, prognosis: dict):
    lines = []
    lines.append("PCOD Health Report")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
    lines.append(f"User: {user}")
    lines.append("")
    lines.append("Inputs:")
    for k, v in data.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Result: {prognosis['level']}")
    lines.append(f"Score: {prognosis['score']}  Confidence: {prognosis['confidence']}%")
    lines.append("")
    lines.append("Diet recommendations (sample): Oats, lean protein, whole grains. Avoid sugary drinks.")
    return "\n".join(lines)

def build_pdf_bytes(user: str, data: dict, prognosis: dict):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(595,842))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40,800,"PCOD Health Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40,785, f"Generated: {datetime.utcnow().isoformat()} UTC")
    pdf.drawString(40,770, f"User: {user}")
    y = 750
    pdf.setFont("Helvetica", 10)
    for k, v in data.items():
        pdf.drawString(40, y, f"{k}: {v}")
        y -= 12
        if y < 80:
            pdf.showPage(); y = 800
    y -= 10
    pdf.drawString(40, y, f"Result: {prognosis['level']} (Score: {prognosis['score']}, Confidence: {prognosis['confidence']}%)")
    pdf.showPage(); pdf.save(); buf.seek(0)
    return buf.read()

# ----------------- CSS (Neo-Glass Purple) -----------------
st.markdown("""
<style>
.header { background: linear-gradient(90deg,#6b21a8,#a78bfa); color:white; padding:12px; border-radius:8px; margin-bottom:12px; }
.card { background: rgba(255,255,255,0.92); padding:12px; border-radius:10px; margin-bottom:10px; box-shadow: 0 6px 18px rgba(0,0,0,0.04); }
.float-btn { position: fixed; right: 28px; bottom: 28px; z-index:9999; background: linear-gradient(90deg,#7c3aed,#c084fc); color:white; border-radius:50%; width:64px; height:64px; font-size:26px; border:none; box-shadow: 0 10px 30px rgba(124,58,237,0.22); }
.small { color:#4b5563; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ----------------- Session init -----------------
if "page" not in st.session_state:
    st.session_state.page = st.experimental_get_query_params().get("page", ["signup"])[0]
if "user" not in st.session_state:
    st.session_state.user = None
if "data" not in st.session_state:
    st.session_state.data = None
if "prognosis" not in st.session_state:
    st.session_state.prognosis = None
if "luna_open" not in st.session_state:
    st.session_state.luna_open = False
if "luna_history" not in st.session_state:
    st.session_state.luna_history = []

# ----------------- Page helpers -----------------
def set_page(page_name: str):
    st.session_state.page = page_name
    st.experimental_set_query_params(page=page_name)
    st.experimental_rerun()



# ----------------- Pages -----------------
def signup_page():
    st.markdown("<h2 style='text-align:center;color:#6b21a8;'>PCOD Classification & Prognosis</h2>", unsafe_allow_html=True)
    st.markdown("<div class='header'><h3 style='margin:0'>Create Account</h3></div>", unsafe_allow_html=True)
    with st.form("signup", clear_on_submit=False):
        username = st.text_input("Username", key="su_user")
        password = st.text_input("Password", type="password", key="su_pass")
        password2 = st.text_input("Confirm Password", type="password", key="su_conf")
        submitted = st.form_submit_button("Create account")
        if submitted:
            if not username or not password:
                st.error("Please enter username and password.")
            elif password != password2:
                st.error("Passwords do not match.")
            elif user_exists(username):
                st.error("Username already exists.")
            else:
                add_user(username, password)
                st.success("Account created. Please login.")
                set_page("login")
    if st.button("Already have an account? Login"):
        set_page("login")


def login_page():
    st.markdown("<h2 style='text-align:center;color:#6b21a8;'>PCOD Classification & Prognosis</h2>", unsafe_allow_html=True)
    st.markdown("<div class='header'><h3 style='margin:0'>Login</h3></div>", unsafe_allow_html=True)
    with st.form("login", clear_on_submit=False):
        username = st.text_input("Username", key="li_user")
        password = st.text_input("Password", type="password", key="li_pass")
        submitted = st.form_submit_button("Login")
        if submitted:
            if verify_user(username, password):
                st.success("Logged in")
                st.session_state.user = username
                set_page("input")
            else:
                st.error("Invalid credentials")
    if st.button("Create new account"):
        set_page("signup")

def input_page():
    if not st.session_state.user:
        st.warning("Please login first.")
        set_page("login")
        return

    st.markdown("<div class='header'><h3 style='margin:0'>Patient Input — Upload Ultrasound (required)</h3></div>", unsafe_allow_html=True)
    with st.form("input_form", clear_on_submit=False):
        age = st.number_input("Age", 10, 80, value=26, key="age")
        height = st.number_input("Height (cm)", 120.0, 220.0, value=160.0, key="height")
        weight = st.number_input("Weight (kg)", 30.0, 200.0, value=60.0, key="weight")
        cycle_irregular = st.checkbox("Irregular periods", value=True, key="cycle_irregular")
        cycle_days = st.number_input("Cycle length (days)", 18, 90, value=28, key="cycle_days")
        ovary_volume = st.number_input("Avg ovary volume (ml)", 1.0, 60.0, value=8.0, key="ovary_volume")
        acne = st.selectbox("Acne", ["No","Yes"], key="acne")
        hair = st.selectbox("Excess hair / hair fall", ["No","Yes"], key="hair")
        stress = st.selectbox("High stress level", ["No","Yes"], key="stress")
        uploaded = st.file_uploader("Ultrasound image (jpg/png) — required", type=["jpg","jpeg","png"], key="uploaded")
        submitted = st.form_submit_button("Analyze")

        if submitted:
            if not uploaded:
                st.error("Ultrasound image is required.")
            else:
                now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                safe_name = f"{st.session_state.user}_{now}_{uploaded.name}"
                path = os.path.join(UPLOAD_DIR, safe_name)
                try:
                    with open(path, "wb") as f:
                        f.write(uploaded.getvalue())
                except Exception as e:
                    st.error(f"Failed to save uploaded file: {e}")
                    return

                # validate format
                img_type = imghdr.what(path)
                if img_type not in ["jpeg","png"]:
                    st.error("Invalid file. Upload a valid ultrasound (JPG/PNG).")
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    return

                try:
                    pil = Image.open(path)

                    # Use the relaxed validator
                    if not is_ultrasound_image(pil):
                        st.error("Uploaded image does not appear to be an ultrasound scan. Please upload a valid ultrasound image (grayscale or grayscale-dominant scan).")
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        return

                    cysts, us_score, debug = analyze_ultrasound_image(pil)

                except Exception as e:
                    st.error(f"Image analysis failed: {e}")
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    return

                data = {
                    "age": age, "height": height, "weight": weight,
                    "bmi": round(weight / ((height/100)**2),1) if height>0 else 0.0,
                    "cycle_irregular": bool(cycle_irregular), "cycle_days": cycle_days,
                    "ovary_volume": ovary_volume, "acne": True if acne=="Yes" else False,
                    "hair": True if hair=="Yes" else False, "stress_high": True if stress=="Yes" else False,
                    "medication": "", "ultrasound_file": path, "cysts": cysts, "us_score": us_score
                }
                prognosis = compute_prognosis(data, us_score)
                st.session_state.data = data
                st.session_state.prognosis = prognosis

                append_user_history(st.session_state.user, {"timestamp": datetime.utcnow().isoformat(), "data": data, "prognosis": prognosis})
                st.success("Analysis complete — opening result")
                set_page("result")
def result_page():
    if not st.session_state.data or not st.session_state.prognosis:
        st.warning("No analysis available. Complete input first.")
        set_page("input")
        return

    data = st.session_state.data
    p = st.session_state.prognosis
    level = p["level"]

    # ================== LEVEL WISE PLANS ==================
    if "Normal" in level or "A" in level:
        lifestyle = [
            "- 25–30 mins brisk walk 5 days/week",
            "- Sleep 7–8 hours nightly",
            "- Hydrate 2–2.5 L/day",
            "- Reduce bakery, sugary snacks slowly",
            "- Light yoga (Sun salutations, breathing)",
        ]
        diet = [
            ("Breakfast","Vegetable oats / idli + sambar","Sugary cereals"),
            ("Mid-morning","Fruit + handful of nuts","Chips"),
            ("Lunch","Whole wheat roti + dal + vegetables","Fried gravies"),
            ("Afternoon","Buttermilk","Milkshakes"),
            ("Dinner","Brown rice/dalia + salad","White rice at night")
        ]

    elif "Moderate" in level or "B" in level:
        lifestyle = [
            "- 35–45 mins walk + twice weekly strength training",
            "- Strict sleep routine (no late nights)",
            "- Hydrate 2.5–3 L/day",
            "- Reduce sugar, pastries, white bread",
            "- Yoga: PCOD-specific asanas",
        ]
        diet = [
            ("Breakfast","Millets + sprouts + eggs","Bread omelette daily"),
            ("Mid-morning","Coconut water + nuts","Tea with sugar"),
            ("Lunch","Quinoa/brown rice + leafy veg","Carbonated drinks"),
            ("Afternoon","Green tea + fruits","Biscuits"),
            ("Dinner","Grilled paneer/chicken + veg","Heavy fried curries")
        ]

    else:
        lifestyle = [
            "- 45–60 mins mix walk + yoga daily",
            "- Consider evening strength/resistance training",
            "- Hydrate 3 L/day min",
            "- Completely avoid refined sugar",
            "- Reduce stress via meditation",
        ]
        diet = [
            ("Breakfast","Moong cheela / oats + eggs","Paratha + butter"),
            ("Mid-morning","Sprouts bowl","Packaged juice"),
            ("Lunch","Millet roti + dal + salad","White rice + sweets"),
            ("Afternoon","Herbal tea + nuts","Cookies"),
            ("Dinner","Soup + steamed veg + light protein","Fried snacks at dinner")
        ]

    st.markdown("<div class='header'><h3 style='margin:0'>PCOD Result</h3></div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    # ---------------- LEFT COLUMN ----------------
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<div style='width:12px;height:12px;border-radius:6px;background:{p['color']}'></div>"
            f"<h4 style='margin:0'>{p['level']}</h4></div>",
            unsafe_allow_html=True
        )
        st.write(f"**Score:** {p['score']}  •  **Confidence:** {p['confidence']}%")
        st.write("**Remarks:**")

        # ---------------- REMARKS BASED ON CYSTS & BMI ----------------
        remarks = []
        if data.get("cysts",0) >= 3:
            remarks.append(f"Follicular-like blobs: {data['cysts']}")
        if data.get("bmi",0) >= 25:
            remarks.append("Elevated BMI")
        if data.get("cycle_irregular"):
            remarks.append("Irregular cycles reported")
        st.write(" • ".join(remarks) if remarks else "No significant remarks.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Lifestyle & Exercise Plan")
        for item in lifestyle:
            st.markdown(item)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT COLUMN ----------------
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Personalized Diet Plan")
        diet_df = pd.DataFrame(diet, columns=["Meal","Recommended","Avoid"])
        st.table(diet_df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Chat with AI Bot")
        st.write("Click the purple floating button (bottom-right) to open the AI chat assistant and clarify doubts.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- 3 COLUMN DETAILS ----------------
    a, b, c = st.columns(3)

    with a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Ultrasound Preview")
        try:
            img = Image.open(data["ultrasound_file"])
            st.image(img, width=240)
            st.write(f"Detected cyst-like blobs: {data.get('cysts',0)}")
            st.write(f"Ultrasound score: {data.get('us_score',0)}/100")
        except Exception:
            st.write("Preview not available")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Supplements (Optional)")
        st.markdown("- Inositol (discuss with clinician)\n- Vitamin D (if deficient)\n- Omega-3 fatty acids")
        st.markdown("</div>", unsafe_allow_html=True)

    with c:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Download Report")
        text = build_text_report(st.session_state.user, data, p)
        pdf = build_pdf_bytes(st.session_state.user, data, p) if REPORTLAB_AVAILABLE else None
        if pdf:
            st.download_button("📄 Download PDF", data=pdf, file_name="pcod_report.pdf", mime="application/pdf")
        else:
            st.download_button("📄 Download Text", data=text, file_name="pcod_report.txt", mime="text/plain")
            st.info("Install reportlab (`pip install reportlab`) to enable PDF export.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- FLOATING AI BOT ----------------
    if st.button("💬 ASK", key="open_Ai_ChatBot"):
        st.session_state.luna_open = not st.session_state.luna_open

    if st.session_state.luna_open:
        st.sidebar.title("💬 ChatBot — Offline Assistant")
        q = st.sidebar.text_input("Ask Bot:")
        if st.sidebar.button("Send"):
            if q and q.strip():
                st.session_state.luna_history.append(("You", q.strip()))
                st.session_state.luna_history.append(("Bot", luna_reply(q.strip(), p)))
                st.experimental_rerun()
        if st.session_state.luna_history:
            for who, msg in st.session_state.luna_history:
                if who == "You":
                    st.sidebar.markdown(f"**You:** {msg}")
                else:
                    st.sidebar.markdown(f"**AI_ChatBot:** {msg}")
        if st.sidebar.button("Close"):
            st.session_state.luna_open = False

    # ---------------- LOGOUT ----------------
    if st.button("Logout"):
        st.session_state.user = None
        set_page("login")


# ----------------- Router (query-param pages) -----------------
page = st.experimental_get_query_params().get("page", [st.session_state.page])[0]

if page == "signup":
    signup_page()
elif page == "login":
    login_page()
elif page == "input":
    input_page()
elif page == "result":
    result_page()
else:
    set_page("signup")
    signup_page()
