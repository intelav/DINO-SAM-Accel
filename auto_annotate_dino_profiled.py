import os
import torch
import torch.profiler
import subprocess
import sys
import auto_annotate_dino_nvtx_optimized as annotator
# ------------- CONFIGURATION -------------
SCRIPT = "auto_annotate_dino_nvtx_optimized.py"     # target script to profile
PROTO_FILE = "./runs/combined_dino_prototypes.json"
TIF = "GC01PS03T0119.tif"                 # pick one test image
DATASET_DIR = "/media/avaish/aiwork/satellite-work/visual_search_dl/dataset/mock-dataset/"
IMAGE_PATH = os.path.join(DATASET_DIR, TIF)
RUNS_DIR = "./runs"
BASENAME = os.path.splitext(TIF)[0]
RUN_DIR = os.path.join(RUNS_DIR, f"{BASENAME}_run")
CAND_DIR = os.path.join(RUN_DIR, "candidates")
os.makedirs(CAND_DIR, exist_ok=True)
CAND_PREFIX = os.path.join(CAND_DIR, f"{BASENAME}_candidates")

SAM_CHECKPOINT = "/media/avaish/aiwork/satellite-work/satellite_annotator/pre-trained-models/sam2_hiera_large.pt"

# ------------- PROFILER -------------
def run_with_profiler():
    """
    Runs the auto_annotate_dino_nvtx pipeline under torch.profiler
    and saves a timeline trace for TensorBoard.
    """
    def trace_handler(p):
        p.export_chrome_trace("trace_auto_annotate.json")
        print("✅ Chrome trace exported → trace_auto_annotate.json")
        p.export_stacks("stacks_auto_annotate.txt", "self_cuda_time_total")
        print("✅ Operator stacks exported → stacks_auto_annotate.txt")

    args = [
        "--proto-file", "./runs/combined_dino_prototypes.json",
        "-i", "/media/avaish/aiwork/satellite-work/visual_search_dl/dataset/mock-dataset/GC01PS03T0119.tif",
        "--band-indexes", "3,2,1",
        "--nir-index", "4",
        "--sam2",
        "--sam-checkpoint",
        "/media/avaish/aiwork/satellite-work/satellite_annotator/pre-trained-models/sam2_hiera_large.pt",
        "--use-masked-feats",
        "--export-candidates", "./runs/GC01PS03T0119_run/candidates/GC01PS03T0119_candidates"
    ]

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        annotator.main(args)
        # # Run the detection pipeline as a subprocess
        # cmd = [
        #     sys.executable, SCRIPT,
        #     "--proto-file", PROTO_FILE,
        #     "-i", IMAGE_PATH,
        #     "--band-indexes", "3,2,1",
        #     "--nir-index", "4",
        #     "--sam2",
        #     "--sam-checkpoint", SAM_CHECKPOINT,
        #     "--use-masked-feats",
        #     "--export-candidates", CAND_PREFIX,
        # ]
        # print(f"🚀 Running subprocess: {' '.join(cmd)}")
        # subprocess.run(cmd, check=True)

    # Print top ops by CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    run_with_profiler()
