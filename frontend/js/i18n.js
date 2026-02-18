(() => {
  const STORAGE_KEY = "ui_lang";
  const SUPPORTED = new Set(["en", "he"]);

  const HE_MAP = {
    "IVR Admin": "מנהל IVR",
    "AI Assistant": "עוזר AI",
    "AI Admin": "מנהל AI",
    "Redirecting to Dashboard...": "מעביר ללוח הבקרה...",
    "Platform": "פלטפורמה",
    "Dashboard": "לוח בקרה",
    "Global Prompt": "פרומפט גלובלי",
    "Restaurants": "מסעדות",
    "Call Logs": "יומני שיחות",
    "System Online": "המערכת פעילה",
    "Platform performance overview": "סקירת ביצועי הפלטפורמה",
    "Active Restaurants": "מסעדות פעילות",
    "Total Calls": "סה״כ שיחות",
    "Missed Calls": "שיחות שלא נענו",
    "Avg Duration": "משך ממוצע",
    "Create Restaurant": "יצירת מסעדה",
    "Restaurant Name": "שם המסעדה",
    "Phone Number": "מספר טלפון",
    "IVR Prompt": "פרומפט IVR",
    "Create Restaurant Agent": "יצירת סוכן מסעדה",
    "Default Prompt (all)": "פרומפט ברירת מחדל (לכולם)",
    "Restaurant List": "רשימת מסעדות",
    "Restaurants": "מסעדות",
    "All Calls": "כל השיחות",
    "Full history of all IVR interactions": "היסטוריה מלאה של כל אינטראקציות ה-IVR",
    "Select a restaurant to manage it": "בחר מסעדה כדי לנהל אותה",
    "Name": "שם",
    "Phone": "טלפון",
    "Save": "שמור",
    "Delete": "מחק",
    "Delete this restaurant? This cannot be undone.": "למחוק את המסעדה? לא ניתן לבטל.",
    "Text Chat": "צ'אט טקסט",
    "Voice Chat": "צ'אט קולי",
    "Start Voice Session": "התחל שיחה קולית",
    "Twilio Webhook": "וובהוק של Twilio",
    "Public Base URL (ngrok/hosted)": "כתובת בסיס ציבורית (ngrok/hosted)",
    "Twilio Webhook URL": "כתובת Webhook של Twilio",
    "Copy": "העתק",
    "Unknown": "לא ידוע",
    "No calls yet": "אין עדיין שיחות",
    "No calls recorded yet": "עדיין לא נרשמו שיחות",
    "No restaurants yet": "אין עדיין מסעדות",
    "Test the AI assistant...": "בדוק את עוזר ה-AI...",
    "Type message": "הקלד הודעה",
    "Send": "שלח",
    "Voice requires HTTPS (or localhost) with microphone access.": "שיחה קולית דורשת HTTPS (או localhost) עם גישת מיקרופון.",
    "Server error, we will fix soon.": "שגיאת שרת, נטפל בזה בקרוב.",
    "Select a restaurant first": "בחר מסעדה קודם",
    "Connection error": "שגיאת חיבור",
    "Status updated": "הסטטוס עודכן",
    "Failed": "נכשל",
    "Delete failed": "המחיקה נכשלה",
    "Restaurant deleted": "המסעדה נמחקה",
    "Restaurant updated": "המסעדה עודכנה",
    "Restaurant name is required": "שם מסעדה הוא שדה חובה",
    "Failed to add restaurant": "הוספת המסעדה נכשלה",
    "Failed to save": "השמירה נכשלה",
    "Global prompt saved successfully": "הפרומפט הגלובלי נשמר בהצלחה",
    "Voice session started": "השיחה הקולית התחילה",
    "Voice session error": "שגיאת שיחה קולית",
    "Copy failed": "ההעתקה נכשלה",
    "Twilio URL copied": "כתובת Twilio הועתקה",
    "You": "אתה",
    "AI": "AI"
  };

  const textOriginals = new WeakMap();

  function getLang() {
    const stored = localStorage.getItem(STORAGE_KEY) || "en";
    return SUPPORTED.has(stored) ? stored : "en";
  }

  function setLang(lang) {
    const next = SUPPORTED.has(lang) ? lang : "en";
    localStorage.setItem(STORAGE_KEY, next);
    applyLanguage(next);
  }

  function t(text) {
    if (getLang() !== "he") return text;
    return HE_MAP[text] || text;
  }

  function translateCoreText(core, lang) {
    if (!core) return core;
    if (lang === "he") return HE_MAP[core] || core;
    return core;
  }

  function translateTextNode(node, lang) {
    if (!textOriginals.has(node)) textOriginals.set(node, node.nodeValue);
    const original = textOriginals.get(node);
    const match = original.match(/^(\s*)(.*?)(\s*)$/s);
    if (!match) return;
    const translated = translateCoreText(match[2], lang);
    node.nodeValue = `${match[1]}${translated}${match[3]}`;
  }

  function translateAttributes(root, lang) {
    const nodes = root.querySelectorAll("[placeholder], [title], input[type='button'], input[type='submit'], button");
    for (const el of nodes) {
      if (el.hasAttribute("placeholder")) {
        const orig = el.dataset.i18nPlaceholder || el.getAttribute("placeholder") || "";
        el.dataset.i18nPlaceholder = orig;
        el.setAttribute("placeholder", translateCoreText(orig, lang));
      }
      if (el.hasAttribute("title")) {
        const orig = el.dataset.i18nTitle || el.getAttribute("title") || "";
        el.dataset.i18nTitle = orig;
        el.setAttribute("title", translateCoreText(orig, lang));
      }
      if (el.tagName === "INPUT" && (el.type === "button" || el.type === "submit")) {
        const orig = el.dataset.i18nValue || el.value || "";
        el.dataset.i18nValue = orig;
        el.value = translateCoreText(orig, lang);
      }
    }
  }

  function translateTree(root, lang = getLang()) {
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const parent = node.parentElement;
      if (!parent) continue;
      const tag = parent.tagName;
      if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") continue;
      if (!node.nodeValue || !node.nodeValue.trim()) continue;
      textNodes.push(node);
    }
    for (const node of textNodes) translateTextNode(node, lang);
    if (root.querySelectorAll) translateAttributes(root, lang);
  }

  function applyLanguage(lang = getLang()) {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "he" ? "rtl" : "ltr";
    if (!document.body) return;
    translateTree(document.body, lang);
    const originalTitle = document.documentElement.dataset.i18nTitle || document.title;
    document.documentElement.dataset.i18nTitle = originalTitle;
    document.title = translateCoreText(originalTitle, lang);
  }

  function addSwitcher() {
    if (document.getElementById("lang-switcher")) return;
    const wrap = document.createElement("div");
    wrap.id = "lang-switcher";
    wrap.style.position = "fixed";
    wrap.style.top = "12px";
    wrap.style.right = "12px";
    wrap.style.zIndex = "10000";
    wrap.style.background = "rgba(0,0,0,0.75)";
    wrap.style.border = "1px solid rgba(255,255,255,0.2)";
    wrap.style.borderRadius = "8px";
    wrap.style.padding = "6px 8px";

    const select = document.createElement("select");
    select.style.background = "transparent";
    select.style.color = "#fff";
    select.style.border = "none";
    select.style.outline = "none";
    select.style.fontSize = "12px";
    select.innerHTML = `
      <option value="en">English</option>
      <option value="he">Hebrew</option>
    `;
    select.value = getLang();
    select.addEventListener("change", (e) => setLang(e.target.value));
    wrap.appendChild(select);
    document.body.appendChild(wrap);
  }

  function installObserver() {
    if (!document.body || document.body.dataset.i18nObserverInstalled) return;
    document.body.dataset.i18nObserverInstalled = "1";
    const observer = new MutationObserver((mutations) => {
      if (getLang() === "en") return;
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) translateTree(node, "he");
          if (node.nodeType === Node.TEXT_NODE) translateTextNode(node, "he");
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  function init() {
    addSwitcher();
    applyLanguage(getLang());
    installObserver();
  }

  window.I18N = {
    init,
    setLang,
    getLang,
    applyLanguage,
    translateTree,
    t
  };
})();
