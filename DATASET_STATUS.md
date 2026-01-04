# Dataset Status

## Current Situation

### Files Available

- **1000 pickle files** in `word-level-dataset-cpu/`
- Files numbered: `00295.pkl` to approximately `01700.pkl`

### JSON Mapping

- **2000 entries** in `filtered_video_to_gloss.json`
- References files from `00295.pkl` to `70371.pkl` (including high-numbered files 60000-70000+)

### Working Vocabulary

- **104 words** with both pickle files AND JSON mappings
- See `filtered_video_to_gloss1.json` for the working subset

## The 104 Available Words

a, abdomen, able, about, above, accent, accident, accomplish, across, active, activity, actor, adapt, address, adjective, adjust, adopt, advanced, advantage, adverb, affect, afraid, africa, after, afternoon, again, against, age, agenda, ago, agree, ahead, aid, aim, airplane, alarm, alcohol, algebra, all, all day, alligator, allow, almost, alone, alphabet, already, also, always, america, amputate, analyze, anatomy, and, angel, angle, angry, animal, anniversary, announce, annoy, another, answer, any, anyway, apart, apartment, appear, appetite, apple, appointment, appreciate, approach, appropriate, approve, april, archery, argue, arizona, arm, army, around, arrest, arrive, art, article, artist, ask, asl, assist, assistant, assume, attention, attitude, attorney, auction, audience, audiologist, audiology, aunt, australia, author, authority, autumn, available

## Missing Components

### Option 1: Get Complete JSON Mapping

- Need: JSON file that maps your 1000 existing pickle files to words
- **896 orphan files** have no word mappings
- Original dataset source: **SignAvatars** (mentioned in project documentation)
- Files exist: `00335.pkl`, `00336.pkl`, etc. (not in current JSON)

### Option 2: Get Missing Pickle Files

- Need: 896 additional pickle files (numbers 60000-70371)
- Would unlock the remaining 1896 words from JSON

## Dataset Source

According to project documentation:

- **SignAvatars dataset** - ASL motion-capture in SMPL-X format
- Each .pkl file contains frame-wise biomechanical parameters:
  - Global orientation
  - Body pose (63 joints)
  - Left-hand pose (45 joints)
  - Right-hand pose (45 joints)
  - Facial & jaw parameters (3 params)

## Public ASL Datasets Found

1. **WLASL** (Word-Level ASL)

   - GitHub: https://github.com/dxli94/WLASL
   - 2000 glosses, video-based
   - Not SMPL-X format (would need conversion)

2. **How2Sign**
   - Website: https://how2sign.github.io/
   - 80+ hours of continuous ASL video
   - Multimodal with 2D/3D skeletons
   - Not SMPL-X format

Note: Neither dataset provides pre-processed SMPL-X pickle files like your current dataset.

## Recommendation

**Proceed with 104 words** while continuing to search for:

1. Complete gloss-to-file mapping for your 1000 pickle files
2. Original SignAvatars dataset source or contact
3. Project team members who may have the complete dataset
