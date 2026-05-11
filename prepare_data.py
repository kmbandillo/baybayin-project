import os
import glob
import shutil
import subprocess
import argparse
import sys


def run(cmd, check=True, capture_output=False, env=None):
    """Run a shell command. cmd can be a list or string."""
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    return subprocess.run(cmd, shell=shell, check=check, capture_output=capture_output, env=env)


def prepare_data(
    source_gt_dir,
    base_dir,
    model_name,
    langdata_dir,
    do_clone=False,
    do_training=False,
    set_dpi=True,
    wordlist_path=None,
):
    print("### Step 0: Configuration ###")
    print(f" source_gt_dir={source_gt_dir}\n base_dir={base_dir}\n model_name={model_name}\n langdata_dir={langdata_dir}\n")

    # Step 1: Optionally clone tesstrain and langdata_lstm
    if do_clone:
        print("### Step 1: Cloning training repositories ###")
        os.makedirs(base_dir, exist_ok=True)
        if not os.path.exists(os.path.join(base_dir, '.git')):
            run(["git", "clone", "https://github.com/tesseract-ocr/tesstrain.git", base_dir])
        if not os.path.exists(langdata_dir):
            run(["git", "clone", "https://github.com/tesseract-ocr/langdata_lstm.git", langdata_dir])
        print("✅ Official training repositories are set up.")

    # Step 2: Sanitize and copy ground truth
    print("### Step 2: Sanitizing ground truth files ###")
    # Place the cleaned ground truth next to the source directory to keep datasets isolated.
    source_abs = os.path.abspath(source_gt_dir)
    source_parent = os.path.dirname(source_abs)
    clean_gt_dir = os.path.join(source_parent, os.path.basename(source_gt_dir) + '_clean')
    os.makedirs(clean_gt_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(source_gt_dir, '*'))
    for source_path in all_files:
        filename = os.path.basename(source_path)
        dest_path = os.path.join(clean_gt_dir, filename)
        try:
            if filename.endswith('.gt.txt'):
                with open(source_path, 'r', encoding='utf-8-sig') as f_in:
                    content = f_in.read()
                with open(dest_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(content)
            else:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, dest_path)
        except Exception:
            print(f"warning: failed to copy {source_path}")
    print("✅ Ground truth files sanitized and copied to:", clean_gt_dir)

    # Step 3: Optionally set DPI on tiff images
    if set_dpi:
        tiff_files = glob.glob(os.path.join(clean_gt_dir, '*.tif')) + glob.glob(os.path.join(clean_gt_dir, '*.tiff'))
        if tiff_files:
            print(f" Setting DPI=300 on {len(tiff_files)} image(s) using mogrify (ImageMagick).")
            try:
                # mogrify -set density 300 <files>
                run(["mogrify", "-set", "density", "300"] + tiff_files)
                print("✅ Image DPI set.")
            except Exception:
                print("warning: mogrify failed or not installed; skipping DPI set.")

    # Step 4: Create Baybayin wordlist from local filipino_latin.txt if present
    print("### Step 3: Creating Baybayin wordlist ###")
    VOWELS = {'a': 'ᜀ', 'i': 'ᜁ', 'u': 'ᜂ'}
    BASE_CONSONANTS = {'b':'ᜊ','k':'ᜃ','d':'ᜇ','g':'ᜄ','h':'ᜑ','l':'ᜎ','m':'ᜋ','n':'ᜈ','ng':'ᜅ','p':'ᜉ','r':'ᜍ','s':'ᜐ','t':'ᜆ','w':'ᜏ','y':'ᜌ'}
    KUDLIT_I, KUDLIT_U, VOWEL_CANCELLER = 'ᜒ', 'ᜓ', '᜔'

    def preprocess(word):
        word = word.lower().replace('ñ','ny').replace('ng','G').replace('c','k').replace('f','p').replace('j','dy')
        word = word.replace('q','k').replace('v','b').replace('x','ks').replace('z','s').replace('G','ng').replace('e','i').replace('o','u')
        return word

    def translate(word):
        baybayin, i, word = "", 0, preprocess(word)
        while i < len(word):
            consonant = 'ng' if i + 1 < len(word) and word[i:i+2] == 'ng' else word[i]
            char_len = 2 if consonant == 'ng' else 1
            if consonant in BASE_CONSONANTS:
                if i + char_len < len(word) and word[i + char_len] in VOWELS:
                    vowel = word[i + char_len]; base = BASE_CONSONANTS[consonant]
                    if vowel == 'i':
                        baybayin += base + KUDLIT_I
                    elif vowel == 'u':
                        baybayin += base + KUDLIT_U
                    else:
                        baybayin += base
                    i += char_len + 1
                else:
                    baybayin += BASE_CONSONANTS[consonant] + VOWEL_CANCELLER
                    i += char_len
            elif consonant in VOWELS:
                baybayin += VOWELS[consonant]
                i += char_len
            else:
                i += 1
        return baybayin

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = []
    if wordlist_path:
        candidate_paths.append(os.path.abspath(wordlist_path))
    else:
        cwd_candidate = os.path.join(os.getcwd(), 'filipino_latin.txt')
        script_candidate = os.path.join(script_dir, 'filipino_latin.txt')
        legacy_candidate = os.path.join(script_dir, 'legacy_offline_training', 'filipino_latin.txt')
        for candidate in (cwd_candidate, script_candidate, legacy_candidate):
            abs_candidate = os.path.abspath(candidate)
            if abs_candidate not in candidate_paths:
                candidate_paths.append(abs_candidate)

    wordlist_src = next((path for path in candidate_paths if os.path.exists(path)), None)
    if not wordlist_src:
        print("warning: filipino_latin.txt not found; skipping wordlist generation.")
    else:
        out_wordlist = f"{model_name}.wordlist"
        print(f" Using wordlist at {wordlist_src}")
        with open(wordlist_src, 'r', encoding='utf-8') as f_in, open(out_wordlist, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                latin_word = line.strip()
                if latin_word:
                    f_out.write(translate(latin_word) + "\n")
        # copy wordlist into langdata_dir if available
        target_langdata_model = os.path.join(langdata_dir, model_name)
        try:
            os.makedirs(target_langdata_model, exist_ok=True)
            shutil.copy2(out_wordlist, target_langdata_model)
            print(f"✅ Baybayin wordlist created and copied to {target_langdata_model}")
        except Exception:
            print(f"warning: failed to copy {out_wordlist} to {target_langdata_model}; you can copy it manually.")

    # Step 5: Organize dataset into tesstrain data folder if base_dir exists
    dest_gt_dir = os.path.join(base_dir, 'data', f"{model_name}-ground-truth")
    if os.path.exists(base_dir):
        os.makedirs(dest_gt_dir, exist_ok=True)
        # rsync-like copy
        for item in os.listdir(clean_gt_dir):
            s = os.path.join(clean_gt_dir, item)
            d = os.path.join(dest_gt_dir, item)
            try:
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            except Exception:
                pass
        try:
            count = len(os.listdir(dest_gt_dir))
        except Exception:
            count = 0
        print(f"✅ All {count} sanitized ground truth files are organized at {dest_gt_dir}.")

    # Step 6: Optionally run training (heavy)
    if do_training:
        print("### Step 6: Running training (this may take many hours) ###")
        cwd = base_dir
        if not os.path.exists(cwd):
            raise RuntimeError(f"base_dir {cwd} does not exist; cannot run training")
        env = os.environ.copy()
        # run make training MODEL_NAME=... MAX_ITERATIONS=30000
        try:
            run(["make", "training", f"MODEL_NAME={model_name}", "MAX_ITERATIONS=30000"], env=env)
        except Exception as e:
            print("error: training failed:", e)

    # Step 7: locate final model
    final_model_path = os.path.join(base_dir, 'data', f"{model_name}.traineddata")
    if os.path.exists(final_model_path):
        print(f"🎉 Your production model is ready at: {final_model_path}")
        try:
            run(["ls", "-lh", final_model_path])
        except Exception:
            pass
    else:
        print("Note: final traineddata not found; training may be incomplete or skipped.")


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare Baybayin training data for tesstrain')
    parser.add_argument('--source', '-s', default=os.path.join(os.getcwd(), 'kaggle_dataset'), help='Path to existing kaggle_dataset ground truth')
    parser.add_argument('--base-dir', '-b', default=os.path.join(os.getcwd(), 'tesseract_training'), help='Base tesstrain directory')
    parser.add_argument('--model-name', '-m', default='bay', help='Model short name')
    parser.add_argument('--langdata-dir', default=None, help='langdata_lstm directory (defaults to <base-dir>/langdata)')
    parser.add_argument('--clone', action='store_true', help='Clone tesstrain and langdata_lstm repos')
    parser.add_argument('--train', action='store_true', help='Run the make training step (very long)')
    parser.add_argument('--no-dpi', action='store_true', help='Do not attempt to set image DPI via mogrify')
    parser.add_argument('--wordlist', default=None, help='Path to filipino_latin.txt for wordlist generation')
    args = parser.parse_args(argv)

    source = args.source
    base_dir = args.base_dir
    langdata_dir = args.langdata_dir or os.path.join(base_dir, 'langdata')

    if not os.path.exists(source):
        print(f"error: source path {source} does not exist. Please point --source to your existing kaggle_dataset ground truth folder.")
        sys.exit(2)

    prepare_data(
        source,
        base_dir,
        args.model_name,
        langdata_dir,
        do_clone=args.clone,
        do_training=args.train,
        set_dpi=not args.no_dpi,
        wordlist_path=args.wordlist,
    )


if __name__ == '__main__':
    main()
