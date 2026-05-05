# MDM Deployment & Dataset Validation Plan

This plan details the exact steps to implement the Motion Diffusion Model (MDM) approach tailored perfectly to your native `how2sign` format, ensuring zero tolerance for data errors when you upload the 30K files.

## 1. Cleanup of Redundant Files
As requested, I will completely remove all files generated during the NSA testing phase, as they are no longer needed.
**Files to be removed:**
- `compile_nsa.py`
- `nsa_unified_pkls/` (Directory)
- `dummy_mapping.json`
- `test_ds.py` & `speed_test.py` & `inspect_how2sign.py`

> [!CAUTION]
> **User Review Required:** Should I also completely delete the `poses/` folder (the 2,318 NSA sequences)? Please confirm if I have permission to delete this 3GB+ folder from your drive, or if you want to keep it just in case.

## 2. Dataset Native Compatibility
I have rigorously inspected your `how2sign-trial` `.pkl` files. 
- They are dictionary files containing multiple keys (e.g., `2d`, `left_valid`, `smplx`).
- The `smplx` key holds the exact `[N, 182]` tensor format our MDM requires.
- The `sign_language_dataset.py` loader I built natively intercepts these keys and uses a custom `CPU_Unpickler` to safely load them even if they were saved on a GPU (which they were). **No data conversion is required.**

## 3. Strict Dataset Readiness Validation ("Zero Tolerance for Problems")
Because training on the A100 is expensive and you cannot tolerate mid-training failures, I will write a highly rigorous validation script (`validate_dataset.py`). 

Before you launch the training job, this script will iterate over all 30,000 files and check:
1. **Corruptions:** Can the file be successfully unpickled?
2. **Schema:** Does `data['smplx']` exist?
3. **Dimensions:** Is the tensor exactly `[N, 182]` or `[N, 169]`? (If 169, our dataloader automatically zero-pads it safely).
4. **NaN/Inf Checks:** Are there any explosive/invalid math values (`NaN` or `Infinity`) in the tensors?
5. **Mapping Alignment:** Does every `.pkl` exist in the JSON mapping, and vice versa?

If any file fails, the script will output a `quarantine_list.txt` so you can instantly remove bad files before they crash the A100.

## 4. Final MDM Implementation Steps
Once the data passes validation, the MDM architecture is fully ready to deploy. I will provide a final `run_training.sh` / batch script configured for the A100.
The training command will simply be:
```bash
python train_diffusion.py train --pkl_dir your_30k_folder --mapping mapping.json --batch_size 128 --epochs 500
```

---

## Open Questions / Approval

Before I execute this cleanup and create the validation engine:

1. **Delete `poses/`?** Can I run a recursive delete on the `poses/` folder to free up space?
2. **Path Name:** Do you plan to name the final 30K dataset folder `how2sign-trial` or something else (e.g., `sign_avatars_30k/`)? I will hardcode the default paths based on your preference.
3. **Approval:** Do you approve of this cleanup and the strict dataset validation approach?
