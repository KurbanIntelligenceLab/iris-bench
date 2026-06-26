"""
Main pipeline: identify both equation form and parameters from video.

Two-stage narrative (no new data needed):
  1. Equation-family selection: A VLM (or a tiny classifier) chooses which ODE
     from a small library applies to the video (e.g. pendulum, dropped_ball, led).
  2. Parameter estimation: The model estimates parameters (alpha, beta, etc.)
     for that chosen ODE.

Use --use_vlm to run Stage 1 with a vision-language model; otherwise dynamics
are inferred from path keywords. Stage 2 is always the same physics model.
"""
from src.models import model as mainmodel
from src import train
from src import loader
import torch
import numpy as np
import argparse
import csv
import os
import matplotlib.pyplot as plt
import matplotlib

plt.ioff()
matplotlib.use('Agg')

video_number = 0

def evaluate_model(model, data_train, dt, name):
    dataloader  = loader.getLoader_folder(data_train, split=False)
    z = None

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model.to(device)

    z = None
    X = []

    for data in dataloader:

        input_Data, out_Data = data

        x0 = input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        x2 = out_Data.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z2_encoder, z2_phys, z3=outputs

        if z is None:
            z = z2_encoder.detach().cpu().numpy()[0][0]
        else:
            z = np.vstack((z,z2_encoder.detach().cpu().numpy()[0][0]))

    
    for i in range(1, z2_encoder.shape[1]):
        z = np.vstack((z,z2_encoder.detach().cpu().numpy()[0][i]))

    alpha = model.pModel.alpha[0].detach().cpu().numpy().item()
    beta = model.pModel.beta[0].detach().cpu().numpy().item()

    print("z2_encoder.shape: ", z2_encoder.shape)
    dt = 1/60
    plt.figure(figsize=(20,5))
    time = np.arange(z.shape[0])
    plt.plot(time*dt, z, label='real', marker='o', linestyle='-')
    #show alpha and beta in text
    plt.text(0.5, 0.5, 'alpha: '+str(alpha)+'\nbeta: '+str(beta), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
 

    plt.xlabel('Time')
    plt.ylabel('z')
    plt.savefig(f'./Results/{name}.png', dpi=300)
    # plt.show()
    plt.close()

    # plt.figure(figsize=(20,5))
    # time = np.arange(z.shape[0])
    # time = time*dt
    # h = 1/z

    # dr = 0.5*time*time*alpha+h[0]
    # plt.plot(time, h, label='real', marker='o', linestyle='-')
    # plt.plot(time, dr, label='dr', marker='o', linestyle='-')
    # #show alpha and beta in text
    # plt.text(0.5, 0.5, 'alpha: '+str(alpha)+'\nmaxtime: '+str(max_time*dt), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    # plt.legend()

    # plt.xlabel('Time')
    # plt.ylabel('z')
    # plt.savefig(f'./Results/{name}_h.png', dpi=300)
    
    # plt.show()
    # plt.close()

    return np.max(z), np.min(z), z[-1].item(), z[0].item()

def execute_experiment(path, dynamics, experiment_name, dt=0.01, loss_name="latent_loss"):

    ''' 
    
    Function to train the model with the data in the path and the dynamics given.
    The function returns the alpha and beta values of the model trained.
    
    Parameters:

        path: str
            Path of the data to train the model.
        dynamics: str
            Dynamics to train the model.
        dt: float
            Time step of the data.
    
    Returns:

        latentEncoder_I: torch model
            Trained model.
        [alpha, beta, max_z, min_z]]: list
            List with the alpha and beta values of the model trained.

    Example:
        
            alpha, beta = execute_experiment('Data/data.npy', 'lorenz', 0.01)

    
    '''

    torch.cuda.empty_cache() 
    torch.manual_seed(42)

    global video_number 

    data_folder = np.load(path, allow_pickle=True)
    # Unwrap 0-d pickle (e.g. object array from allow_pickle=True)
    if data_folder.ndim == 0 and data_folder.dtype == object:
        data_train = data_folder.item()
    else:
        data_train = data_folder
    if not isinstance(data_train, np.ndarray):
        data_train = np.array(data_train)

    # Normalize shape: expect (N, nf, 1, H, W) e.g. (N, 10, 1, 56, 100)
    if data_train.ndim == 1:
        nf, h, w = 10, 56, 100
        expected_per_sample = nf * 1 * h * w
        if data_train.size % expected_per_sample != 0:
            raise ValueError(
                f"Loaded array is 1D with size {data_train.size}; "
                f"expected multiple of {expected_per_sample} (nf=10, h=56, w=100). "
                "Re-save the .npy with shape (N, 10, 1, 56, 100) or use video2npy."
            )
        n = data_train.size // expected_per_sample
        data_train = data_train.reshape(n, nf, 1, h, w).astype(np.float32)

    if data_train.ndim < 4:
        raise ValueError(
            f"Expected .npy with shape (N, 10, 1, H, W); got ndim={data_train.ndim}, shape={getattr(data_train, 'shape', '?')}. "
            "Use video2npy to convert videos to the expected format."
        )

    if data_train.shape[0] == 0:
        raise ValueError(
            "Loaded .npy has zero samples (shape[0]==0). "
            "video2npy needs at least 10 frames per video (with adaptive step). "
            "Re-run video2npy on your source videos so short clips get samples (video2npy now uses smaller step for videos with 20–59 frames)."
        )

    # Subsample when very large (e.g. long pendulum clips) to keep training time reasonable
    MAX_SAMPLES_PER_FILE = 1500
    n_orig = data_train.shape[0]
    if n_orig > MAX_SAMPLES_PER_FILE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_orig, size=MAX_SAMPLES_PER_FILE, replace=False)
        data_train = data_train[idx]
        print(f"Subsampled to {MAX_SAMPLES_PER_FILE} samples (was {n_orig}).")

    sample_frames = data_train[0, 0, :, :]

    # plot and save the first frame
    results_dir = os.path.join('./Results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    for i in range(1, data_train.shape[1]):
        sample_frames = np.hstack((sample_frames, data_train[0, i, :, :]))
    
    sample_frames = sample_frames.transpose(1,2,0)
    plt.figure(figsize=(5,25))
    plt.imshow(sample_frames, cmap= 'gray')
    plt.savefig(os.path.join(results_dir, 'sample_frame.png'), dpi=300)
    # plt.show()
    plt.close()
  
    print("Data shape: ", data_train.shape)
    print("Data range: \nMin: ", np.min(data_train), "\nMax: ", np.max(data_train)) 
    
    # Fixed batch size of 128 for all runs (matches the reported experiments).
    batch_size = 128

    train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader_folder(data_train, split=True, batch_size = batch_size)


    #define the model

    latentEncoder_I = mainmodel.EndPhys(dt = dt,
                                    pmodel = dynamics,
                                    init_phys = 10.0, 
                                    initw=True)

    #train model (always returns model, log, [best_a, best_b]; on NaN/early exit no checkpoint is saved)
    latentEncoder_I, log, params = train.train(latentEncoder_I,
                                               train_dataloader,
                                               test_dataloader,
                                               lr_phys=0.01,
                                               loss_name=loss_name,
                                               experiment_name=experiment_name)

    alpha_log = []
    if log and "alpha" in log[0]:
        alpha_log.append([element["alpha"] for element in log])
        plt.figure(figsize=(10,5))
        plt.plot(alpha_log[0], label='alpha')
        plt.xlabel('Epoch')
        plt.ylabel('Alpha')
        plt.legend()
        plt.savefig(f'./Results/{experiment_name}/alpha_log_{video_number}.png', dpi=300)
        plt.close()

    best_model_path = f'./Results/{experiment_name}/best_model.pt'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=True)
        latentEncoder_I.load_state_dict(checkpoint)
        alpha = latentEncoder_I.pModel.alpha[0].detach().cpu().numpy().item()
        beta = latentEncoder_I.pModel.beta[0].detach().cpu().numpy().item()
    else:
        alpha, beta = params[0], params[1]

    max_z, min_z, z0, z1 = evaluate_model(latentEncoder_I, data_train, dt, experiment_name+'/'+str(video_number))

    return latentEncoder_I, [alpha, beta, max_z, min_z, z0, z1]

def get_dynamics(path):
    ''' 
    Function to get the dynamics of the model trained.
    
    Parameters:
        path (str): The file path to check for specific dynamics keywords.

    Returns:
        str: The dynamic type found in the path, if any, else None.
    '''	
    # Order matters: more specific (IRIS) before substrings (e.g. dropping_ball before dropped_ball)
    dynamics_keywords = [
        'two_moving_pendulum_one_static', 'two_moving_pendulums',
        'dropping_ball', 'falling_ball', 'sliding_cone', 'hitting_cones', 'rotation',
        'pendulum', 'sliding_block', 'bouncing_ball', 'dropped_ball', 'led', 'free_fall', 'torricelli',
    ]

    # Normalize the path to ignore case and spaces
    normalized_path = path.replace(' ', '').lower()

    for keyword in dynamics_keywords:
        if keyword in normalized_path:
            print(f"Found dynamics keyword: {keyword}")
            return keyword
    
    return None

def _normalize_gt_for_vlm(gt):
    """Map path keyword to VLM choice for evaluation (e.g. bouncing_ball -> dropped_ball)."""
    if gt == "bouncing_ball":
        return "dropped_ball"
    return gt


def iterate_folders_and_process(root_folder, output_folder='output', dt=0.01, use_vlm=False, use_vlm_improved=False, loss_name='latent_loss'):

    global video_number 

    # Initialize a list to collect data
    results = []
    # VLM evaluation records (path, gt_from_path, pred_from_vlm) when use_vlm=True
    vlm_records = []

    # Iterate through all folders in the root directory
    for folder_name, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(folder_name, file)
                
                # Extract folder path components relative to the root folder
                relative_folder_path = os.path.relpath(folder_name, root_folder)
                path_components = relative_folder_path.split(os.sep)

                print(f"Processing {file_path}...")

                # -------------------------------------------------------------------------
                # Stage 1: Equation-family selection (which ODE from the library applies?)
                # VLM or path-based keyword selects dynamics; no new data needed.
                # -------------------------------------------------------------------------
                gt_from_path = get_dynamics(file_path)
                gt_norm = _normalize_gt_for_vlm(gt_from_path) if gt_from_path else None

                try:
                    experiment = '_'.join(component.replace(' ', '') for component in path_components)
                    if use_vlm:
                        if use_vlm_improved:
                            from src.utils.vlm_improved import detect_dynamics_from_npy
                        else:
                            from src.utils.vlm_dynamics import detect_dynamics_from_npy
                        pred_vlm = detect_dynamics_from_npy(file_path)
                        if pred_vlm is None:
                            current_dynamics = gt_from_path
                            print("VLM returned no result; using path-based dynamics:", current_dynamics)
                        else:
                            current_dynamics = pred_vlm
                            print(f"VLM detected dynamics: {current_dynamics}")
                        if gt_norm is not None:
                            vlm_records.append({"path": file_path, "gt": gt_norm, "pred": pred_vlm or ""})
                    else:
                        current_dynamics = gt_from_path
                        print(f"Current dynamics: {current_dynamics}")
                    if current_dynamics is None:
                        raise ValueError(f"Could not determine dynamics for {file_path}. Use --use_vlm or ensure path contains a known dynamics keyword.")

                    # -------------------------------------------------------------------------
                    # Stage 2: Parameter estimation for the chosen ODE
                    # -------------------------------------------------------------------------
                    video_number += 1
                    model, [a, b, max_z, min_z, z0, z1] = execute_experiment(file_path, current_dynamics, output_folder, dt, loss_name=loss_name)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    a, b, max_z, min_z, z0, z1 = 0,0,0,0,0,0

                results.append(path_components + [a, b, max_z, min_z, z0, z1])

    # VLM measurable contribution: accuracy and confusion matrix in the same output folder
    if use_vlm and vlm_records:
        results_dir = os.path.join("./Results", output_folder)
        os.makedirs(results_dir, exist_ok=True)
        gt_list = [r["gt"] for r in vlm_records]
        pred_list = [r["pred"] if r["pred"] else "<empty>" for r in vlm_records]
        correct = sum(1 for r in vlm_records if r["pred"] == r["gt"])
        accuracy = correct / len(vlm_records)
        vlm_label = "VLM improved (enhanced prompt + 5 frames)" if use_vlm_improved else "VLM dynamics selection (pipeline)"
        with open(os.path.join(results_dir, "vlm_accuracy.txt"), "w") as f:
            f.write(f"{vlm_label}\n")
            f.write(f"Total: {len(vlm_records)}  Correct: {correct}  Accuracy: {accuracy:.2%}\n")
        with open(os.path.join(results_dir, "vlm_results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "gt", "pred"])
            w.writeheader()
            w.writerows(vlm_records)
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            labels = sorted(set(gt_list) | set(pred_list))
            if "<empty>" in labels:
                labels.remove("<empty>")
                labels.append("<empty>")
            cm = confusion_matrix(gt_list, pred_list, labels=labels)
            with open(os.path.join(results_dir, "vlm_confusion_matrix.csv"), "w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow([""] + labels)
                for i, label in enumerate(labels):
                    cw.writerow([label] + list(cm[i]))
            with open(os.path.join(results_dir, "vlm_accuracy.txt"), "a") as f:
                f.write("\nConfusion matrix (rows=GT, cols=pred):\n")
                f.write(str(labels) + "\n")
                f.write(str(cm) + "\n")
                f.write("\n" + classification_report(gt_list, pred_list, labels=[l for l in labels if l != "<empty>"], zero_division=0))
        except ImportError:
            pass
        print(f"VLM evaluation: accuracy {accuracy:.2%}  -> Results/{output_folder}/vlm_accuracy.txt, vlm_results.csv, vlm_confusion_matrix.csv")

    # Write results to a CSV file
    if not results:
        print("No results collected. Possible causes: no .npy files under the path, path missing dynamics keywords (e.g. pendulum, dropped_ball, sliding_block, led, free_fall, torricelli), or every file raised an error. Check that the data path (e.g. ./delfys75) exists and contains subfolders whose names include the dynamics type.")
        return
    max_depth = max(len(row) - 2 for row in results)  # Determine max depth of folder structure
    #headers = [f'Folder_Level_{i+1}' for i in range(max_depth)] + ['alpha', 'beta', 'max_z', 'min_z']
    headers = ['run', 'alpha', 'beta', 'max_z', 'min_z',  'z0', 'z1']

    if not os.path.exists(f'./Results/{output_folder}'):
        os.makedirs(f'./Results/{output_folder}')

    with open(f'./Results/{output_folder}/{output_folder}.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)  # Header row
        for row in results:
            padded_row = row[:-2] + [''] * (max_depth - len(row[:-2])) + row[-2:]
            writer.writerow(padded_row)
    
    # mean_a, std_a, mean_b, std_b  = get_mean_std_from_csv('./Results/'+output_csv)

    # print(f"Mean alpha: {mean_a}, Std alpha: {std_a}")
    # print(f"Mean beta: {mean_b}, Std beta: {std_b}")

    print(f"Data successfully written to {output_folder}")

def main():
    '''
    Main function to execute the experiment.

    Parameters:

        None

    Returns:

        None

    Example:

    python main.py --path Data/data.npy --experiment_name experiment1 --dynamics lorenz --dt 0.01

    '''
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU in use: {gpu_name}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU found — training will use CPU.")
        print("To use an NVIDIA GPU: install PyTorch with CUDA, e.g.")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  (see https://pytorch.org/get-started/locally/ for your CUDA version)")
        

    parser = argparse.ArgumentParser(description="Required parameter for a single experiment")
    #Example 
    #python main.py --path Data/data.npy --experiment_name experiment1 --dynamics lorenz --dt 0.01

    # Adding arguments
    
    parser.add_argument("--dt", type=str, required=True, help="Delta time")
    parser.add_argument("--path", type=str, required=True, help="Data path")
    parser.add_argument("--outfolder", type=str, default="output.csv", help="Output CSV file name")
    parser.add_argument("--use_vlm", action="store_true", help="Stage 1: Use VLM (OpenRouter) to select equation family from video. Otherwise use path keywords. Set OPENROUTER_API_KEY.")
    parser.add_argument("--vlm_improved", action="store_true", help="Use improved VLM (enhanced prompt + 5 frames). Implies --use_vlm. Saves metrics to same output folder for comparison.")
    parser.add_argument("--loss", type=str, default="latent_loss", choices=["latent_loss", "latent_loss_multistep"], help="Loss: latent_loss (1-step) or latent_loss_multistep (weighted 1..5 step).")
    args = parser.parse_args()

    use_vlm = args.use_vlm or args.vlm_improved
    use_vlm_improved = args.vlm_improved

    # Parse dt, accepting either a decimal ("0.05") or a fraction ("1/60").
    # Avoids eval() on user input.
    dt_str = args.dt.strip()
    if "/" in dt_str:
        num, den = dt_str.split("/", 1)
        dt = float(num) / float(den)
    else:
        dt = float(dt_str)

    # Call your function with the specified or default output file name
    iterate_folders_and_process(args.path, output_folder=args.outfolder, dt=dt, use_vlm=use_vlm, use_vlm_improved=use_vlm_improved, loss_name=args.loss)

def get_mean_std_from_csv(csv_file):
    '''
    Function to get the mean and standard deviation of the data in the csv file.

    Parameters:

        csv_file: str
            Path of the csv file.
    
    Returns:

        mean: float
            Mean of the data in the csv file.
        std: float
            Standard deviation of the data in the csv file.

    Example:

        mean, std = get_mean_std_from_csv('output.csv')

    '''
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    a = data[:,1]
    b = data[:,2]
    mean_a = np.mean(data[:,1])
    std_a = np.std(data[:,1])
    mean_b = np.mean(data[:,2])
    std_b = np.std(data[:,2])

    return mean_a, std_a, mean_b, std_b

if __name__ == "__main__":

    '''
    Main function to execute the experiment.

    Parameters:
    
            None

    Returns:
    
                None
    Example:

    python main.py --path Data/data.npy --experiment_name experiment1 --dynamics lorenz --dt 0.01
    CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv /home/acastanedagarc/Projects/Vphy/Vphy/container_vphys.sif python /home/acastanedagarc/Projects/Vphy/Vphy/main.py --path /home/acastanedagarc/Projects/data/ --dt 0.1 >> output.log 2>&1
    '''
    
    torch.cuda.empty_cache() 
    torch.manual_seed(42)
    main()
