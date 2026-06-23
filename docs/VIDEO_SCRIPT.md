# ARIA-Enforce — Demo / Pitch Video Script (~2.5 min)

Record on the machine that has the **vehicle model** (your main PC) with the
**helmet `best.pt`** dropped in, so violations actually fire. Screen-record the
browser + a voiceover. Keep it tight.

---

### 0:00–0:20 — Hook (talking head or title slide)
> "Traffic cameras produce more images every day than any officer can review.
> ARIA-Enforce turns a single traffic photo into a court-ready challan —
> automatically. Built for Indian roads."

Show slide 1 of the deck.

### 0:20–0:45 — The problem (deck slides 2)
> "Manual review is slow and inconsistent, and Indian traffic — helmetless
> riders, triple-riding, mixed vehicles, rain and low light — breaks generic
> tools. We trained on 11 Indian vehicle classes plus a dedicated helmet model."

### 0:45–1:30 — Live demo: detection (screen-record the app, Tab 1)
1. Run `python -m aria_enforce.app`, open the URL.
2. Upload a photo of a helmetless rider. Click **Detect violations**.
> "Upload an image — the system enhances it, detects road users, flags the
> violation, reads the plate, and scores severity."
3. Point at the **annotated evidence** image and the **e-challan slip**.
> "And it doesn't stop at detection — it issues an e-challan with the Motor
> Vehicles Act section, the fine, and the evidence. Detect to enforce."
4. Upload a second image (illegal parking / stop-line) to show multiple types.

### 1:30–1:55 — Analytics + prediction (Tabs 2 & 3)
> "Every challan is stored and searchable, with trends and daily summaries."
Show Tab 2. Then Tab 3:
> "And a predictive layer over 8,000 real traffic events flags congestion
> hotspots before they peak."

### 1:55–2:25 — The bigger picture (deck slide 8)
> "ARIA-Enforce is one agent of a full adaptive-traffic platform — the same
> backend that flags a red-light runner can re-time the signal with our RL
> controller. Enforcement and flow optimisation in one system."

### 2:25–2:40 — Close (deck slide 10)
> "From a single photo to a court-ready challan — accurate, automated, and built
> to scale. ARIA-Enforce."

---

**Tips:** 1080p, record at a calm pace, show real outputs (not slides) for the
demo section, and keep total length under any limit the form specifies. Upload to
YouTube (unlisted) or Google Drive and paste the link in the **Video URL** field.
