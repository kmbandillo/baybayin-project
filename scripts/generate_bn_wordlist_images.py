import random
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
WORDLIST = BASE / 'dataset/training_wordlist.txt'
OUTPUT_DIR = BASE / 'dataset/words/withnoise'
FONTS_DIR = BASE / 'baybayin_dataset'
LATIN_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_NAME = 'Baybayin Namin'
NUM_IMAGES = 200
WORDS_PER_LINE = (3, 6)
PTSIZE = '64'
LEADING = '72'
RESOLUTION = '300'
CROP_MARGIN = 20
LATIN_SIZE = 72
LATIN_OFFSET = 40
EXTRA_BOTTOM = LATIN_SIZE + 2 * LATIN_OFFSET
THRESHOLD = 200

KUDLIT_I = 'ᜒ'
KUDLIT_U = 'ᜓ'
VIRAMA = '᜔'
VOWEL_MAP = {'ᜀ': 'a', 'ᜁ': 'i', 'ᜂ': 'o'}
VOWEL_REVERSE = {v: k for k, v in VOWEL_MAP.items()}
CONSONANT_MAP = {
    'ᜊ': 'b','ᜃ': 'k','ᜄ': 'g','ᜅ': 'ng','ᜆ': 't','ᜇ': 'd','ᜍ': 'r','ᜈ': 'n','ᜉ': 'p','ᜋ': 'm','ᜌ': 'y','ᜎ': 'l','ᜏ': 'w','ᜐ': 's','ᜑ': 'h'
}
CONSONANT_REVERSE = {v: k for k, v in CONSONANT_MAP.items()}

random.seed(42)

def encode_for_font(text):
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in CONSONANT_MAP:
            base = CONSONANT_MAP[ch]
            nxt = text[i + 1] if i + 1 < len(text) else ''
            if nxt == KUDLIT_I:
                result.append(base + 'i'); i += 2
            elif nxt == KUDLIT_U:
                result.append(base + 'o'); i += 2
            elif nxt == VIRAMA:
                result.append(base + '='); i += 2
            else:
                result.append(base + 'a'); i += 1
        elif ch in VOWEL_MAP:
            result.append(VOWEL_MAP[ch]); i += 1
        else:
            result.append(ch); i += 1
    return ' '.join(''.join(result).split())

def romanize(text):
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == ' ':
            result.append(' '); i += 1; continue
        if ch in CONSONANT_MAP:
            base = CONSONANT_MAP[ch]
            nxt = text[i + 1] if i + 1 < len(text) else ''
            if nxt == KUDLIT_I:
                result.append(base + 'i'); i += 2
            elif nxt == KUDLIT_U:
                result.append(base + 'o'); i += 2
            elif nxt == VIRAMA:
                result.append(base); i += 2
            else:
                result.append(base + 'a'); i += 1
        elif ch in VOWEL_MAP:
            result.append(VOWEL_MAP[ch]); i += 1
        else:
            result.append(ch); i += 1
    roman = ''.join(result).upper()
    return ' '.join(roman.split())

def find_components(subarr):
    h, w = subarr.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or subarr[r, c] >= THRESHOLD:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            min_r = max_r = r
            min_c = max_c = c
            area = 0
            while stack:
                rr, cc = stack.pop()
                if subarr[rr, cc] >= THRESHOLD:
                    continue
                area += 1
                min_r = min(min_r, rr)
                max_r = max(max_r, rr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for nr in (rr-1, rr, rr+1):
                    for nc in (cc-1, cc, cc+1):
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and subarr[nr, nc] < THRESHOLD:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            comps.append({'min_r': min_r, 'max_r': max_r, 'min_c': min_c, 'max_c': max_c, 'area': area})
    return comps

def choose_component(comps, kind):
    if not comps:
        return None
    if kind == 'top':
        return min(comps, key=lambda c: c['min_r'])
    if kind == 'bottom':
        return max(comps, key=lambda c: c['max_r'])
    if kind == 'right':
        return max(comps, key=lambda c: c['max_c'])
    return comps[0]

def render_phrase(index, baybayin_text):
    base_name = OUTPUT_DIR / f'wordlist_bn_{index:04d}'
    encoded = encode_for_font(baybayin_text)
    latin = romanize(baybayin_text)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as tmp:
        tmp.write(encoded + '\n')
        tmp_path = Path(tmp.name)
    cmd = [
        'text2image',
        f'--fonts_dir={FONTS_DIR}',
        f'--font={FONT_NAME}',
        f'--text={tmp_path}',
        f'--outputbase={base_name}',
        f'--ptsize={PTSIZE}',
        f'--leading={LEADING}',
        f'--resolution={RESOLUTION}',
        '--max_pages=1'
    ]
    subprocess.run(cmd, check=True)
    tmp_path.unlink(missing_ok=True)
    tif_path = base_name.with_suffix('.tif')
    box_path = base_name.with_suffix('.box')
    img = Image.open(tif_path).convert('L')
    arr = np.array(img)
    coords = np.argwhere(arr < THRESHOLD)
    if coords.size == 0:
        raise RuntimeError('No foreground pixels found')
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    min_r = max(min_r - CROP_MARGIN, 0)
    min_c = max(min_c - CROP_MARGIN, 0)
    max_r = min(max_r + CROP_MARGIN, arr.shape[0]-1)
    max_c = min(max_c + CROP_MARGIN, arr.shape[1]-1)
    crop = arr[min_r:max_r+1, min_c:max_c+1]
    height_cropped, width_cropped = crop.shape
    lines = box_path.read_text(encoding='utf-8').splitlines()
    entries = []
    for raw_line in lines:
        if not raw_line.strip():
            continue
        parts = raw_line.split(' ', 1)
        label = parts[0]
        if len(parts) < 2:
            continue
        rest = parts[1].strip()
        coords = rest.split()
        if len(coords) < 5:
            continue
        x1, y1, x2, y2, page = map(int, coords[:5])
        img_top = arr.shape[0] - y2
        img_bottom = arr.shape[0] - y1
        crop_top = max(0, img_top - min_r)
        crop_bottom = max(0, img_bottom - min_r)
        crop_x1 = max(0, x1 - min_c)
        crop_x2 = max(0, x2 - min_c)
        crop_bottom = min(height_cropped, crop_bottom)
        crop_x2 = min(width_cropped, crop_x2)
        subarr = crop[crop_top:crop_bottom, crop_x1:crop_x2]
        comps = find_components(subarr)
        if not comps:
            if label == '':
                base_coords = (crop_x1, crop_top, crop_x2, crop_bottom)
            else:
                continue
        else:
            comps.sort(key=lambda c: c['area'], reverse=True)
            base_comp = comps[0]
            base_coords = (
                crop_x1 + base_comp['min_c'],
                crop_top + base_comp['min_r'],
                crop_x1 + base_comp['max_c'] + 1,
                crop_top + base_comp['max_r'] + 1,
            )
        if label == '':
            entries.append((' ', base_coords, page))
            continue
        suffix = label[-1]
        base_key = label[:-1].lower()
        if base_key in CONSONANT_REVERSE:
            entries.append((CONSONANT_REVERSE[base_key], base_coords, page))
            kind = None
            diac_char = None
            if suffix in ('i', 'I'):
                kind = 'top'; diac_char = KUDLIT_I
            elif suffix in ('o', 'O'):
                kind = 'right'; diac_char = KUDLIT_U
            elif suffix == '=':
                kind = 'bottom'; diac_char = VIRAMA
            if diac_char:
                others = comps[1:]
                if others:
                    diac_comp = choose_component(others, kind)
                    if diac_comp:
                        diac_coords = (
                            crop_x1 + diac_comp['min_c'],
                            crop_top + diac_comp['min_r'],
                            crop_x1 + diac_comp['max_c'] + 1,
                            crop_top + diac_comp['max_r'] + 1,
                        )
                        entries.append((diac_char, diac_coords, page))
        elif label in VOWEL_REVERSE:
            entries.append((VOWEL_REVERSE[label], base_coords, page))
    cropped_img = Image.fromarray(crop)
    new_height = height_cropped + EXTRA_BOTTOM
    total_img = Image.new('L', (width_cropped, new_height), color=255)
    total_img.paste(cropped_img, (0, EXTRA_BOTTOM))
    draw = ImageDraw.Draw(total_img)
    font = ImageFont.truetype(LATIN_FONT, LATIN_SIZE)
    text_bbox = draw.textbbox((0, 0), latin, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x = (width_cropped - text_w) // 2
    text_y = (EXTRA_BOTTOM - text_h) // 2
    draw.text((text_x, text_y), latin, font=font, fill=0)
    png_path = base_name.with_suffix('.png')
    total_img.save(png_path)
    tif_path.unlink(missing_ok=True)
    box_lines = []
    for idx_entry, (char, (x1, top, x2, bottom), page) in enumerate(entries):
        y1 = height_cropped - bottom + EXTRA_BOTTOM
        y2 = height_cropped - top + EXTRA_BOTTOM
        box_lines.append(f"{char} {x1} {y1} {x2} {y2} {page}")
    box_path.write_text('\n'.join(box_lines) + '\n', encoding='utf-8')
    gt_path = base_name.with_suffix('.gt.txt')
    gt_path.write_text(baybayin_text + '\n', encoding='utf-8')

def main():
    words = [line.strip() for line in WORDLIST.read_text(encoding='utf-8').splitlines() if line.strip()]
    for idx in range(1, NUM_IMAGES + 1):
        count = random.randint(*WORDS_PER_LINE)
        phrase = ' '.join(random.choice(words) for _ in range(count))
        render_phrase(idx, phrase)
        print(f'Generated wordlist_bn_{idx:04d}')

if __name__ == '__main__':
    main()
