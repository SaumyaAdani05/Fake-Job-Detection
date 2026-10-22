# UI / UX Specification

**Style direction:** Clean and trustworthy. Light theme, security-tool feel, calm blues and greens, plain language.
**Framework:** Streamlit (custom theme + light CSS). No custom JS framework.

---

## 1. Design Principles

1. **Calm, not scary.** The tool deals with scams, but the UI should feel reassuring and clear, not alarmist.
2. **Plain language.** No ML jargon on the main flow. A first-year student or a parent should understand every sentence.
3. **One main action per screen.** Paste, click, result.
4. **Explain everything.** Never show a verdict without reasons.
5. **Honest about uncertainty.** Show a confidence level and a disclaimer. The tool gives a probability, not a legal ruling.
6. **Never rely on colour alone.** Always pair colour with an icon and a text label.

## 2. Colour Palette

| Role | Hex | Use |
|------|-----|-----|
| Primary | `#1D4ED8` | Buttons, links, headings accent |
| Primary hover | `#1E40AF` | Button hover |
| Background | `#F8FAFC` | Page background |
| Surface | `#FFFFFF` | Cards, input areas |
| Text | `#0F172A` | Body text |
| Muted text | `#475569` | Helper text, captions |
| Border | `#E2E8F0` | Card and input borders |
| Safe / Real | `#16A34A` | "Likely Real" verdict |
| Warning / Suspicious | `#D97706` | "Suspicious" verdict |
| Danger / Fake | `#DC2626` | "Likely Fake" verdict |

Contrast: body text on background must meet WCAG AA (4.5:1). Check verdict colours on their tinted backgrounds.

## 3. Typography

- Font: **Inter** (fallback: system sans-serif). Streamlit's default sans is acceptable if Inter is not loaded.
- Page title: 32 px, semi-bold. Section headings: 20 to 24 px. Body: 16 px. Captions: 14 px.
- Line height around 1.5 for body text.

## 4. Layout and Navigation

Use Streamlit's multipage structure or a top-level tab/segmented control:

1. **Check a Job** (home, default)
2. **How It Works** (About)

Keep the sidebar minimal: project name, short privacy note, link to GitHub repo.

### Page 1: Check a Job

```
-------------------------------------------------------------
 [shield icon]  Fake Job Post Checker
 Paste a job posting and we'll tell you if it looks suspicious.
-------------------------------------------------------------
 [ Quick Paste ]  [ Detailed Form ]          <- tabs

 Quick Paste tab:
 +---------------------------------------------------------+
 | Paste the full job posting here...                      |
 |                                                         |
 +---------------------------------------------------------+
 [ Check this job ]   [ Try an example v ]

 Privacy note: Your text is not saved or stored.
-------------------------------------------------------------
 RESULT (appears after click)
 +---------------------------------------------------------+
 |  [icon] LIKELY FAKE            Scam probability: 87%    |
 |  [=========================------]                      |
 +---------------------------------------------------------+
 Why was this flagged?
  - [!] No company profile or logo was provided
  - [!] The post asks for a registration fee
  - [i] Words like "urgent" and "no experience needed" are common in scam posts
 What you should do
  - Never pay money to get a job
  - Verify the company on its official website
  - ...
 Disclaimer: This tool gives an estimate, not proof.
-------------------------------------------------------------
```

### Detailed Form tab
- Two-column layout on desktop, single column on mobile.
- Group fields: **Basics** (title, location, department, employment type), **Company** (company profile, industry, has logo), **Job details** (description, requirements, benefits), **Pay and extras** (salary range, remote, screening questions).
- Helper text under uncertain fields (for example: "Leave blank if the post doesn't mention it. Missing information is itself a useful signal.").
- Same "Check this job" button and same result panel as Quick Paste.

### Page 2: How It Works
- Short intro: what the tool does and why.
- Simple 3-step graphic or list: *Read the post, look for patterns, estimate scam probability*.
- Dataset in one paragraph (EMSCAD, about 17,880 postings, fake ones are rare).
- Models tried and which one is live.
- Metrics table: Recall, Precision, F1, ROC-AUC (static values from the training run).
- Limitations section (dataset age, English only, estimates not proof).
- Team and course credit.

## 5. Components

| Component | Behaviour |
|-----------|-----------|
| Verdict card | Coloured border/tint, icon, label, probability %. Three states: Likely Real, Suspicious, Likely Fake |
| Probability bar | Horizontal progress bar; colour matches verdict |
| Reason list | Each item has an icon (warning/info/check) and one plain sentence. Max 6 shown, most important first |
| Example loader | Dropdown with "Sample real post" and "Sample fake post" that fills the text box |
| Spinner | "Analysing the post..." shown during prediction |
| Disclaimer box | Muted background, small text, always visible under the result |

## 6. Copy and Tone

- Verdict wording: **Likely Real**, **Suspicious**, **Likely Fake**. Never "Guaranteed scam" or "100% safe".
- Reasons are full sentences, in friendly language.
- Avoid: "feature importance", "TF-IDF", "logit", "class imbalance" on the main page.
- Error messages are helpful and specific, for example: "That looks too short. Paste at least a few lines of the job post."

## 7. States to Design

| State | What the user sees |
|-------|--------------------|
| Empty (first visit) | Instructions, example button, privacy note |
| Loading | Spinner with short text |
| Success | Verdict card, reasons, next steps |
| Input too short | Inline warning, no prediction |
| Very long input | Truncate safely, show note "Only the first N characters were analysed" |
| Unexpected error | Friendly message, suggest retry, no stack trace |
| Cold start (Render) | Page may take 30 to 60 s to wake. Mention in README, not in UI |

## 8. Responsive and Accessibility Checklist

- [ ] Usable at 360 px width (phone)
- [ ] Buttons at least 44 px tall
- [ ] Every verdict has icon + text + colour
- [ ] Text contrast passes WCAG AA
- [ ] Form fields have visible labels (no placeholder-only labels)
- [ ] Works with keyboard navigation
- [ ] No auto-playing animations

## 9. Do and Don't (UI)

**Do**
- Keep the first screen focused: title, one text box, one button.
- Show reasons with every verdict.
- Show the privacy note near the input.
- Use consistent spacing and rounded cards.

**Don't**
- Don't use red-heavy, alarming layouts or skull/warning-siren imagery.
- Don't display raw probabilities like `0.8731` without context.
- Don't show model internals (coefficients, token ids) to end users.
- Don't hide the disclaimer.
- Don't add pop-ups, sign-up walls, or ads.
- Don't pre-fill the text box with real company data.

## 10. Streamlit Theme (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#1D4ED8"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0F172A"
font = "sans serif"

[server]
headless = true
```
