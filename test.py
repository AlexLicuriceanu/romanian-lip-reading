from talknet.main import run_talknet_batch
import os

if __name__ == "__main__":
    CLIP_DIR = "/home/rhc/licenta4/trimmed_outputs/Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live"
    SAVE_DIR = "./asd_outputs/"
    MAX_PARALLEL_JOBS = 2  # Adjust based on your GPU capacity

    run_talknet_batch(CLIP_DIR, SAVE_DIR, max_workers=MAX_PARALLEL_JOBS, in_memory_threshold=300)