# IVR AI System

## Run Instructions (Urdu/Hinglish)

1. Sab se pehle **Laragon** open karein.
2. Laragon me **MySQL** start karein.
3. Database banayein: `ivr_ai_system`
4. Ab backend folder me jaa kar `app.py` run karein:

```powershell
cd backend
python app.py
```

5. Phir frontend server start karein:

```powershell
cd frontend
python -m http.server 8000
```

6. Browser me ye URL open karein: `http://localhost:8000/dashboard.html`
7. Dashboard ke through aap system use kar sakte hain.
8. Note (local): frontend `8000` par chalta hai aur API calls backend `http://localhost:5050` par jati hain, isliye backend process running hona chahiye.

## Admin Auth

Dashboard access ke liye admin login required hai. `.env` me ye set karein:

```
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=your-strong-password
TOKEN_TTL_HOURS=24
```

`TOKEN_TTL_HOURS` optional hai. Default 24 hours hai.

### Restaurant Admin Assignment

Owner (super admin) per-restaurant admins create/assign kar sakta hai. UI me "Assign Admin" card aur restaurant list ke saath 👤 icon se assignment hoti hai.
Security note: passwords kabhi UI me show nahi kiye ja sakte (hash hotay hain). Agar password bhool jayein, naya password set karke user ko re-create karein.

## Admin Login

Email/password auth enabled hai. `.env` me ye variables add karein:

```env
ADMIN_EMAIL="admin@example.com"
ADMIN_PASSWORD="change-this-password"
TOKEN_TTL_HOURS=24
```

Dashboard open karte hi login screen ayegi. Same creds se sign in karein.

## Frontend Server (VPS)

### Frontend start command

```bash
cd /opt/ivr_ai_system/frontend
nohup python3 -m http.server 8080 --bind 0.0.0.0 > frontend.log 2>&1 &
```

### Kya same command dobara chalana sahi hai?

Blindly nahi.

Agar already running hai to duplicate server start hoga aur port busy ho jayega.

Pehle check karo:

```bash
lsof -i :8080
```

Agar output aaye to server already running hai.

Old process stop karo:

```bash
pkill -f http.server
```

Phir frontend start command dobara chalao.

### Frontend logs kaise dekho

Live logs:

```bash
tail -f /opt/ivr_ai_system/frontend/frontend.log
```

Last 50 lines:

```bash
tail -n 50 /opt/ivr_ai_system/frontend/frontend.log
```

## VPS Troubleshooting (Service + Logs)

### STEP 3 — reload systemd

Service edit ke baad:

```bash
systemctl daemon-reload
systemctl restart ivr_ai_system
systemctl status ivr_ai_system
```

### STEP 4 — agar phir bhi fail

Manual run karo:

```bash
cd /opt/ivr_ai_system/backend
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

### LAST error logs (sabse important)

Ye command run karo:

```bash
journalctl -u ivr_ai_system.service -n 50 --no-pager
```

Isme last 50 lines ayengi jahan Python crash hua.

### LIVE logs dekhne (real time)

Service start karo aur live error dekho:

```bash
journalctl -u ivr_ai_system.service -f
```

Phir dusri terminal me:

```bash
systemctl restart ivr_ai_system
```

Jo bhi error ayega live dikhega.

### full logs (agar upar se samajh na aaye)

```bash
journalctl -u ivr_ai_system.service
```
