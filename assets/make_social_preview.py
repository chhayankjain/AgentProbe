"""Render the 1280x640 GitHub social preview (drawn at 2x, then downsampled).

Uses macOS system fonts (Helvetica Neue, Menlo). Run:
    python assets/make_social_preview.py assets/social-preview.png
then upload the PNG under Settings > General > Social preview.
"""
import sys
from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 1280 * S, 640 * S
HN, MENLO = "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Menlo.ttc"

def font(path, size, idx):
    return ImageFont.truetype(path, size * S, index=idx)

BG_TOP, BG_BOT = (10, 14, 22), (17, 22, 34)
TEXT, MUTED, DIM = (240, 244, 250), (160, 170, 186), (110, 120, 138)
GREEN, AMBER, BLUE, PURPLE, ORANGE = (63, 185, 80), (230, 170, 60), (110, 170, 255), (190, 140, 255), (255, 160, 110)
CARD, CARD_EDGE, TERM = (22, 27, 36), (48, 54, 66), (13, 17, 24)

img = Image.new("RGB", (W, H))
d = ImageDraw.Draw(img)
for y in range(H):  # vertical gradient
    t = y / H
    d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT)))
# soft green glow behind the card
glow = Image.new("RGB", (W, H), (0, 0, 0))
gd = ImageDraw.Draw(glow)
for r in range(420 * S, 0, -6 * S):
    k = int(14 * (1 - r / (420 * S)))
    gd.ellipse([960 * S - r, 330 * S - r, 960 * S + r, 330 * S + r], fill=(0, k, int(k * 0.5)))
px = img.load()
g = glow.load()
for y in range(0, H):
    for x in range(W // 2, W):
        a, b = px[x, y], g[x, y]
        if b != (0, 0, 0):
            px[x, y] = (min(255, a[0] + b[0]), min(255, a[1] + b[1]), min(255, a[2] + b[2]))
d = ImageDraw.Draw(img)

def T(xy, text, f, fill):
    d.text((xy[0] * S, xy[1] * S), text, font=f, fill=fill)

# ---- left: brand, headline, pitch, install --------------------------------
L = 64
d.rounded_rectangle([L * S, 58 * S, (L + 14) * S, 72 * S], radius=3 * S, fill=GREEN)
T((L + 24, 52), "AgentProbe", font(HN, 24, 1), TEXT)

hl = font(HN, 52, 1)
for i, line in enumerate(["Will your agent", "survive a", "broken tool?"]):
    T((L, 128 + i * 62), line, hl, TEXT if i < 2 else GREEN)

sub = font(HN, 22, 0)
for i, line in enumerate(["Deterministic tool faults in pytest.", "The test fails unless every", "fault actually fired."]):
    T((L, 336 + i * 31), line, sub, MUTED)

pill_f = font(MENLO, 19, 1)
label = "pip install agentprobe-toolkit"
tw = d.textlength(label, font=pill_f) / S
d.rounded_rectangle([L * S, 490 * S, (L + tw + 36) * S, 536 * S], radius=10 * S, fill=(20, 40, 28), outline=(40, 110, 60), width=2 * S)
T((L + 18, 501), label, pill_f, GREEN)
T((L, 562), "github.com/chhayankjain/agentprobe-toolkit  ·  MIT", font(HN, 17, 0), DIM)

# ---- right: code card + test output ---------------------------------------
cx0, cy0, cx1, cy1 = 590, 70, 1224, 574
d.rounded_rectangle([cx0 * S, cy0 * S, cx1 * S, cy1 * S], radius=16 * S, fill=CARD, outline=CARD_EDGE, width=2 * S)
for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    d.ellipse([(cx0 + 22 + i * 22) * S, (cy0 + 20) * S, (cx0 + 34 + i * 22) * S, (cy0 + 32) * S], fill=c)
T((cx0 + 96, cy0 + 16), "test_agent.py", font(MENLO, 15, 0), DIM)
d.line([cx0 * S, (cy0 + 50) * S, cx1 * S, (cy0 + 50) * S], fill=CARD_EDGE, width=2 * S)

mono, mono_b = font(MENLO, 15, 0), font(MENLO, 15, 1)
KW, FN, STR, PLAIN, COMMENT = PURPLE, BLUE, ORANGE, (220, 226, 236), DIM
code = [
    [("from ", KW), ("agentprobe.testing ", PLAIN), ("import ", KW), ("probe", PLAIN)],
    [],
    [("def ", KW), ("test_survives_a_timeout", FN), ("():", PLAIN)],
    [("    with ", KW), ("probe() ", PLAIN), ("as ", KW), ("p:", PLAIN)],
    [("        tools = p.", PLAIN), ("wrap", FN), ("([payment_status])", PLAIN)],
    [("        p.", PLAIN), ("on", FN), ("(", PLAIN), ('"payment_status"', STR), (").", PLAIN), ("fail_first", FN), ("(1)", PLAIN)],
    [("        r = p.", PLAIN), ("run", FN), ("(agent(tools), ", PLAIN), ('"Order 42?"', STR), (")", PLAIN)],
    [("        r.", PLAIN), ("assert_answer_contains", FN), ("(", PLAIN), ('"confirmed"', STR), (")", PLAIN)],
]
y = cy0 + 70
for line in code:
    x = cx0 + 26
    for text, col in line:
        T((x, y), text, mono, col)
        x += d.textlength(text, font=mono) / S
    y += 25

# terminal strip
ty0 = 352
d.rectangle([(cx0 + 2) * S, ty0 * S, (cx1 - 2) * S, (cy1 - 2) * S], fill=TERM)
d.rounded_rectangle([(cx0 + 2) * S, (cy1 - 40) * S, (cx1 - 2) * S, (cy1 - 2) * S], radius=14 * S, fill=TERM)
d.line([cx0 * S, ty0 * S, cx1 * S, ty0 * S], fill=CARD_EDGE, width=2 * S)
out = [
    [("$ ", DIM), ("pytest", PLAIN)],
    [("[exercised] ", GREEN), ("payment_status: ", PLAIN), ("timeout", AMBER), (" on call 1", PLAIN)],
    [("[pass]      ", GREEN), ("answer contains 'confirmed'", PLAIN)],
    [("[inferred]  ", BLUE), ("agent retried payment_status once", PLAIN)],
    [("gate PASSED", GREEN), ("  ·  1 passed in 0.04s", MUTED)],
    [("agent skips the tool? ", DIM), ("not_exercised -> FAILED", (248, 113, 113))],
]
y = ty0 + 18
for line in out:
    x = cx0 + 26
    for text, col in line:
        f = mono_b if col in (GREEN,) else mono
        T((x, y), text, f, col)
        x += d.textlength(text, font=f) / S
    y += 26

img = img.resize((1280, 640), Image.LANCZOS)
img.save(sys.argv[1], optimize=True)
print("saved", sys.argv[1])
