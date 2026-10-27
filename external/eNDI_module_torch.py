

import time
import numpy as np
import torch

from FANSI_module_torch import compute_xsim_torch, _build_xsim_kernels

def ndi(params):
    """
    
    Nonlinear Dipole Inversion. Gradient Descent solver.
    
    Parameters: params - structure with
    Required fields:
    params.input: local field map, in radians.
    params.K: dipole kernel in the frequency space.
    Optional fields:
    params.alpha: regularization weight (use small values for stability).
    params.maxOuterIter: maximum number of iterations.
    params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
    params.tau: gradient descent rate.
    params.precond: preconditionate solution (for stability).
    params.mask: (optional) binary mask for NRMSE calculation. If not provided,
                 params.weight (squared) is used as mask.
    params.chi_cosmos: (optional) ground truth susceptibility map, used for NRMSE tracking.
    params.isLcurve: (optional) boolean to compute residual and regularization norms for L-curve analysis.
    params.isXSIM: (optional) boolean to compute XSIM at each iteration (default: False).
    params.scale: (optional) scale factor to bring x / chi_cosmos into the ~1 ppm
                  range expected by the XSIM constants, before computing XSIM.

    Output: out - structure with the following fields:
    out.x: calculated susceptibility map, in radians.
    out.iter: number of iterations.
    out.time: total elapsed time (including pre-calculations).
    out.update_history: list of relative update (%) values per iteration.
    out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
    out.xsim_history: (if chi_cosmos is provided and isXSIM) list of XSIM values per iteration.
    out.res_norm: (if isLcurve is True) residual norm of the final solution.
    out.reg_norm: (if isLcurve is True) L2 norm of chi, corresponding to the alpha regularizer.

    """
    start_time = time.time()
    # Required parameters
    kernel = torch.tensor(params["K"], dtype=torch.float32)
    phase = torch.tensor(params["input"], dtype=torch.float64)

    alpha = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float)
    tau = torch.tensor(params.get("tau", 2.0), dtype=torch.float)
    N = phase.shape
    num_iter = params.get("maxOuterIter", 150)

    weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
    weight = weight * weight

    isPrecond = params.get("isPrecond", False)

    if isPrecond:
        x = torch.tensor(params.get("precond", weight * phase), dtype=torch.float32)
    else:
        x = torch.zeros(N, dtype=torch.float64)

    # Load the separate mask for NRMSE calculation
    mask_np = params.get("mask", None)
    if mask_np is None:
        print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
        mask_for_rmse = weight  # Fallback to using the (squared) fidelity weight
    else:
        mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32)

    # --- L-curve toggle added here ---
    is_lcurve = params.get("isLcurve", False)

    # --- XSIM toggle added here ---
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("scale", 1.0), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = kernel.to(device)
    phase = phase.to(device)
    x = x.to(device)
    alpha = alpha.to(device)
    tau = tau.to(device)
    weight = weight.to(device)
    mask_for_rmse = mask_for_rmse.to(device)
    scale = scale.to(device)

    Kt = torch.conj(kernel)

    # --- NRMSE History Initialization ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
        masked_true = mask_for_rmse * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8
    # --- End NRMSE Initialization ---

    # --- XSIM History Initialization ---
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask_for_rmse.bool()
    xsim_history = []
    # --- End XSIM Initialization ---

    regTime = time.time()
    update_history = []
    for t in range(num_iter):
        # Update x: susceptibility estimate
        x_prev = x
        phix = susc2field(kernel, x)
        loss = susc2field(Kt, weight * torch.sin(phix - phase)) - alpha * x
        x = x_prev - tau * loss

        # --- NRMSE Calculation ---
        if chi_cosmos_tensor is not None:
            masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # --- XSIM Calculation ---
            if isXSIM:
                mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_history.append(mssim.item())
            # --- End XSIM Calculation ---
        # --- End NRMSE Calculation ---

        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())

        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_history[-1]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')

    end_time = time.time()
    regTime = end_time - regTime
    total_time = end_time - start_time

    print(f'Regularization Time: {regTime:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')

    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        phix = susc2field(kernel, x)
        res_norm = torch.sum(weight * torch.abs(torch.exp(1j * phix) - torch.exp(1j * phase))**2)
        reg_norm = torch.sum(x**2)
    # --- End L-curve Calculations ---

    out = {"x": x.cpu().numpy().astype(np.float32), "iter": t+1, "time": total_time, "update_history": update_history}

    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_history

    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["reg_norm"] = reg_norm.item()

    return out


def ndi_auto(params):
    """
    
    Nonlinear Dipole Inversion. Gradient Descent solver with an automatic stopping criterion.
    
    Parameters: params - structure with
    Required fields:
    params.input: local field map, in radians.
    params.K: dipole kernel in the frequency space.
    Optional fields:
    params.alpha: regularization weight (use small values for stability).
    params.maxOuterIter: maximum number of iterations.
    params.voxelSize: spatial resolution.
    params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
    params.tau: gradient descent rate.
    params.precond: preconditionate solution (for stability).
    params.mask: (optional) binary mask for NRMSE calculation. If not provided,
                 params.weight (squared) is used as mask.
    params.chi_cosmos: (optional) ground truth susceptibility map, used for NRMSE tracking.
    params.isLcurve: (optional) boolean to compute residual and regularization norms for L-curve analysis.
    params.isXSIM: (optional) boolean to compute XSIM at each iteration (default: False).
    params.scale: (optional) scale factor to bring x / chi_cosmos into the ~1 ppm
                  range expected by the XSIM constants, before computing XSIM.

    Output: out - structure with the following fields:
    out.x: calculated susceptibility map, in radians.
    out.iter: number of iterations.
    out.time: total elapsed time (including pre-calculations).
    out.regTime: elapsed time (excluding pre-calculations).
    out.update_history: list of relative update (%) values per iteration.
    out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
    out.xsim_history: (if chi_cosmos is provided and isXSIM) list of XSIM values per iteration.
    out.res_norm: (if isLcurve is True) residual norm of the final solution.
    out.reg_norm: (if isLcurve is True) L2 norm of chi, corresponding to the alpha regularizer.

    """
    start_time = time.time()
    # Required parameters
    kernel = torch.tensor(params["K"], dtype=torch.float32)
    phase = torch.tensor(params["input"], dtype=torch.float64)

    alpha = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float32)
    voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
    tau = torch.tensor(params.get("tau", 2.0), dtype=torch.float)
    N = phase.shape
    num_iter = params.get("maxOuterIter", 150)

    weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
    weight = weight * weight

    isPrecond = params.get("isPrecond", False)

    if isPrecond:
        x = torch.tensor(params.get("precond", weight * phase), dtype=torch.float32)
    else:
        x = torch.zeros(N, dtype=torch.float64)

    m1, m2 = create_freqmasks_auto(voxelSize, kernel)
    m1 = torch.tensor(m1, dtype=torch.float64)
    m2 = torch.tensor(m2, dtype=torch.float64)

    # Load the separate mask for NRMSE calculation
    mask_np = params.get("mask", None)
    if mask_np is None:
        print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
        mask_for_rmse = weight  # Fallback to using the (squared) fidelity weight
    else:
        mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32)

    # --- L-curve toggle added here ---
    is_lcurve = params.get("isLcurve", False)

    # --- XSIM toggle added here ---
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("scale", 1.0), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = kernel.to(device)
    phase = phase.to(device)
    x = x.to(device)
    alpha = alpha.to(device)
    tau = tau.to(device)
    weight = weight.to(device)
    m1 = m1.to(device)
    m2 = m2.to(device)
    mask_for_rmse = mask_for_rmse.to(device)
    scale = scale.to(device)

    # --- NRMSE History Initialization ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
        masked_true = mask_for_rmse * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8
    # --- End NRMSE Initialization ---

    # --- XSIM History Initialization ---
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask_for_rmse.bool()
    xsim_history = []
    # --- End XSIM Initialization ---

    regTime = time.time()
    update_history = []
    for t in range(num_iter):
        # Update x: susceptibility estimate
        x_prev = x
        phix = susc2field(kernel, x)
        loss = susc2field(torch.conj(kernel), weight * torch.sin(phix - phase)) - alpha * x
        x = x_prev - tau * loss

        # --- NRMSE Calculation ---
        if chi_cosmos_tensor is not None:
            masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # --- XSIM Calculation ---
            if isXSIM:
                mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_history.append(mssim.item())
            # --- End XSIM Calculation ---
        # --- End NRMSE Calculation ---

        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())

        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_history[-1]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')

        e1, e2 = compute_freqe_auto(x, m1, m2)

        if e1 > e2 and t > 3:
            print('Automatic stop reached!')
            break

    end_time = time.time()
    regTime = end_time - regTime
    total_time = end_time - start_time
    print(f'Regularization Time: {regTime:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')

    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        phix = susc2field(kernel, x)
        res_norm = torch.sum(weight * torch.abs(torch.exp(1j * phix) - torch.exp(1j * phase))**2)
        reg_norm = torch.sum(x**2)
    # --- End L-curve Calculations ---

    out = {"x": x.cpu().numpy().astype(np.float32), "iter": t+1, "time": total_time, "update_history": update_history}

    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_history

    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["reg_norm"] = reg_norm.item()

    return out


def ndiCG(params):
    """
    
    Nonlinear Dipole Inversion. Conjugate Gradient Descent solver.
    
    Parameters: params - structure with
    Required fields:
    params.input: local field map, in radians.
    params.K: dipole kernel in the frequency space.
    Optional fields:
    params.alpha: regularization weight (use small values for stability).
    params.maxOuterIter: maximum number of iterations.
    params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
    params.precond: preconditionate solution (for stability).
    params.mask: (optional) binary mask for NRMSE calculation. If not provided,
                 params.weight (squared) is used as mask.
    params.chi_cosmos: (optional) ground truth susceptibility map, used for NRMSE tracking.
    params.isLcurve: (optional) boolean to compute residual and regularization norms for L-curve analysis.
    params.isXSIM: (optional) boolean to compute XSIM at each iteration (default: False).
    params.scale: (optional) scale factor to bring x / chi_cosmos into the ~1 ppm
                  range expected by the XSIM constants, before computing XSIM.

    Output: out - structure with the following fields:
    out.x: calculated susceptibility map, in radians.
    out.iter: number of iterations.
    out.time: total elapsed time (including pre-calculations).
    out.regTime: elapsed time (excluding pre-calculations).
    out.update_history: list of relative update (%) values per iteration (including the initial line-search step).
    out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
    out.xsim_history: (if chi_cosmos is provided and isXSIM) list of XSIM values per iteration.
    out.res_norm: (if isLcurve is True) residual norm of the final solution.
    out.reg_norm: (if isLcurve is True) L2 norm of chi, corresponding to the alpha regularizer.

    """
    start_time = time.time()
    # Required parameters
    kernel = torch.tensor(params["K"], dtype=torch.float32)
    phase = torch.tensor(params["input"], dtype=torch.float64)

    alpha = torch.tensor(params.get("alpha", 1E-6), dtype=torch.float)
    N = phase.shape
    num_iter = params.get("maxOuterIter", 150)

    weight = torch.tensor(params.get("weight", torch.ones(N)), dtype=torch.float32)
    weight = weight * weight

    isPrecond = params.get("isPrecond", False)

    if isPrecond:
        x = torch.tensor(params.get("precond", weight * phase), dtype=torch.float32)
    else:
        x = torch.zeros(N, dtype=torch.float64)

    # Load the separate mask for NRMSE calculation
    mask_np = params.get("mask", None)
    if mask_np is None:
        print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
        mask_for_rmse = weight  # Fallback to using the (squared) fidelity weight
    else:
        mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32)

    # --- L-curve toggle added here ---
    is_lcurve = params.get("isLcurve", False)

    # --- XSIM toggle added here ---
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("scale", 1.0), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase = phase.to(device)
    kernel = kernel.to(device)
    x = x.to(device)
    weight = weight.to(device)
    alpha = alpha.to(device)
    mask_for_rmse = mask_for_rmse.to(device)
    scale = scale.to(device)

    # --- NRMSE History Initialization ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
        masked_true = mask_for_rmse * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8
    # --- End NRMSE Initialization ---

    # --- XSIM History Initialization ---
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask_for_rmse.bool()
    xsim_history = []
    # --- End XSIM Initialization ---

    regTime = time.time()
    update_history = []
    # Initiate with a gradient descent, with automatic step by line search.
    phix = susc2field(kernel, x)
    dx = -susc2field(torch.conj(kernel), weight * torch.sin(phix - phase)) - alpha * x

    B = dx * dx
    A = dx * (susc2field(torch.conj(kernel), weight * torch.sin(susc2field(kernel, dx))) + alpha * dx)
    tau = torch.sum(B) / (torch.sum(A) + 1e-8)

    x_prev = x
    x = x_prev + tau * dx
    x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
    update_history.append(x_update.item())

    # --- NRMSE Calculation ---
    if chi_cosmos_tensor is not None:
        masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
        norm_error = torch.norm(masked_error)
        nrmse = 100 * norm_error / norm_true
        rmse_history.append(nrmse.item())

        # --- XSIM Calculation ---
        if isXSIM:
            mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
            xsim_history.append(mssim.item())
        # --- End XSIM Calculation ---
    # --- End NRMSE Calculation ---

    if chi_cosmos_tensor is not None:
        if isXSIM:
            print(f'Iter: 0   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_history[-1]:.4f}')
        else:
            print(f'Iter: 0   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
    else:
        print(f'Iter: 0   Update: {x_update:.6f}')

    s = dx
    # Start CG steps
    for t in range(num_iter):
        # update x: susceptibility estimate
        x_prev = x
        phix = susc2field(kernel, x)
        dx_prev = dx
        dx = -susc2field(torch.conj(kernel), weight * torch.sin(phix - phase)) - alpha * x

        betaPR = max(torch.sum(dx * (dx - dx_prev)) / (torch.sum(dx * dx) + 1e-8), 0)  # Automatic reset

        s = dx + betaPR * s

        B = s * dx
        A = s * (susc2field(torch.conj(kernel), weight * torch.sin(susc2field(kernel, s))) + alpha * s)
        tau = torch.sum(B) / (torch.sum(A) + 1e-8)

        x = x_prev + tau * s

        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())

        # --- NRMSE Calculation ---
        if chi_cosmos_tensor is not None:
            masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # --- XSIM Calculation ---
            if isXSIM:
                mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_history.append(mssim.item())
            # --- End XSIM Calculation ---
        # --- End NRMSE Calculation ---

        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_history[-1]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')

    end_time = time.time()
    regTime = end_time - regTime
    total_time = end_time - start_time
    print(f'Regularization Time: {regTime:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')

    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        phix = susc2field(kernel, x)
        res_norm = torch.sum(weight * torch.abs(torch.exp(1j * phix) - torch.exp(1j * phase))**2)
        reg_norm = torch.sum(x**2)
    # --- End L-curve Calculations ---

    out = {"x": x.cpu().numpy().astype(np.float32), "iter": t+1, "time": total_time, "update_history": update_history}

    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_history

    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["reg_norm"] = reg_norm.item()

    return out


def ndiCG_auto(params):
    """
    
    Nonlinear Dipole Inversion. Conjugate Gradient Descent solver.
    
    Parameters: params - structure with
    Required fields:
    params.input: local field map, in radians.
    params.K: dipole kernel in the frequency space.
    Optional fields:
    params.alpha: regularization weight (use small values for stability).
    params.maxOuterIter: maximum number of iterations.
    params.voxelSize: spatial resolution.
    params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
    params.precond: preconditionate solution (for stability).
    params.mask: (optional) binary mask for NRMSE calculation. If not provided,
                 params.weight (squared) is used as mask.
    params.chi_cosmos: (optional) ground truth susceptibility map, used for NRMSE tracking.
    params.isLcurve: (optional) boolean to compute residual and regularization norms for L-curve analysis.
    params.isXSIM: (optional) boolean to compute XSIM at each iteration (default: False).
    params.scale: (optional) scale factor to bring x / chi_cosmos into the ~1 ppm
                  range expected by the XSIM constants, before computing XSIM.

    Output: out - structure with the following fields:
    out.x: calculated susceptibility map, in radians.
    out.iter: number of iterations.
    out.time: total elapsed time (including pre-calculations).
    out.regTime: elapsed time (excluding pre-calculations).
    out.update_history: list of relative update (%) values per iteration (including the initial line-search step).
    out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
    out.xsim_history: (if chi_cosmos is provided and isXSIM) list of XSIM values per iteration.
    out.res_norm: (if isLcurve is True) residual norm of the final solution.
    out.reg_norm: (if isLcurve is True) L2 norm of chi, corresponding to the alpha regularizer.

    """
    start_time = time.time()
    # Required parameters
    kernel = torch.tensor(params["K"], dtype=torch.float32)
    phase = torch.tensor(params["input"], dtype=torch.float64)

    alpha = torch.tensor(params.get("alpha", 1E-6), dtype=torch.float)
    voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
    N = phase.shape
    num_iter = params.get("maxOuterIter", 150)

    weight = torch.tensor(params.get("weight", torch.ones(N)), dtype=torch.float32)
    weight = weight * weight

    isPrecond = params.get("isPrecond", False)

    if isPrecond:
        x = torch.tensor(params.get("precond", weight * phase), dtype=torch.float32)
    else:
        x = torch.zeros(N, dtype=torch.float64)

    m1, m2 = create_freqmasks_auto(voxelSize, kernel)
    m1 = torch.tensor(m1, dtype=torch.float64)
    m2 = torch.tensor(m2, dtype=torch.float64)

    # Load the separate mask for NRMSE calculation
    mask_np = params.get("mask", None)
    if mask_np is None:
        print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
        mask_for_rmse = weight  # Fallback to using the (squared) fidelity weight
    else:
        mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32)

    # --- L-curve toggle added here ---
    is_lcurve = params.get("isLcurve", False)

    # --- XSIM toggle added here ---
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("scale", 1.0), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase = phase.to(device)
    kernel = kernel.to(device)
    x = x.to(device)
    weight = weight.to(device)
    alpha = alpha.to(device)
    m1 = m1.to(device)
    m2 = m2.to(device)
    mask_for_rmse = mask_for_rmse.to(device)
    scale = scale.to(device)

    # --- NRMSE History Initialization ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
        masked_true = mask_for_rmse * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8
    # --- End NRMSE Initialization ---

    # --- XSIM History Initialization ---
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask_for_rmse.bool()
    xsim_history = []
    # --- End XSIM Initialization ---

    regTime = time.time()
    update_history = []
    # Initiate with a gradient descent, with automatic step by line search.
    phix = susc2field(kernel, x)
    dx = -susc2field(torch.conj(kernel), weight * torch.sin(phix - phase)) - alpha * x

    B = dx * dx
    A = dx * (susc2field(torch.conj(kernel), weight * torch.sin(susc2field(kernel, dx))) + alpha * dx)
    tau = torch.sum(B) / (torch.sum(A) + 1e-8)

    x_prev = x
    x = x_prev + tau * dx
    x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
    update_history.append(x_update.item())

    # --- NRMSE Calculation ---
    if chi_cosmos_tensor is not None:
        masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
        norm_error = torch.norm(masked_error)
        nrmse = 100 * norm_error / norm_true
        rmse_history.append(nrmse.item())

        # --- XSIM Calculation ---
        if isXSIM:
            mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
            xsim_history.append(mssim.item())
        # --- End XSIM Calculation ---
    # --- End NRMSE Calculation ---

    if chi_cosmos_tensor is not None:
        if isXSIM:
            print(f'Iter: 0   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_history[-1]:.4f}')
        else:
            print(f'Iter: 0   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
    else:
        print(f'Iter: 0   Update: {x_update:.6f}')

    s = dx
    # Start CG steps
    for t in range(num_iter):
        # update x: susceptibility estimate
        x_prev = x
        phix = susc2field(kernel, x)
        dx_prev = dx
        dx = -susc2field(torch.conj(kernel), weight * torch.sin(phix - phase)) - alpha * x

        betaPR = max(torch.sum(dx * (dx - dx_prev)) / (torch.sum(dx * dx) + 1e-8), 0)  # Automatic reset

        s = dx + betaPR * s

        B = s * dx
        A = s * (susc2field(torch.conj(kernel), weight * torch.sin(susc2field(kernel, s))) + alpha * s)
        tau = torch.sum(B) / (torch.sum(A) + 1e-8)

        x = x_prev + tau * s

        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())

        # --- NRMSE Calculation ---
        if chi_cosmos_tensor is not None:
            masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # --- XSIM Calculation ---
            if isXSIM:
                mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_history.append(mssim.item())
            # --- End XSIM Calculation ---
        # --- End NRMSE Calculation ---

        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_history[-1]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')

        e1, e2 = compute_freqe_auto(x, m1, m2)

        if e1 > e2 and t > 3:
            print('Automatic stop reached!')
            break

    end_time = time.time()
    regTime = end_time - regTime
    total_time = end_time - start_time
    print(f'Regularization Time: {regTime:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')

    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        phix = susc2field(kernel, x)
        res_norm = torch.sum(weight * torch.abs(torch.exp(1j * phix) - torch.exp(1j * phase))**2)
        reg_norm = torch.sum(x**2)
    # --- End L-curve Calculations ---

    out = {"x": x.cpu().numpy().astype(np.float32), "iter": t+1, "time": total_time, "update_history": update_history}

    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_history

    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["reg_norm"] = reg_norm.item()

    return out


def susc2field(D, X):
    """
    
    Susceptibility to Field calculation
    
    """
    phi = torch.real(torch.fft.ifftn(D * torch.fft.fftn(X)))
    
    return phi


def create_freqmasks_auto(spatial_res, kernel):
    """
    
    Calculates the masks that define Regions of Interest in the Fourier domain of
    reconstructed QSM images. These masks are defined by boundares that depend on:
    1) Dipole kernel coefficients.
    2) Absolute frequency radial range.
    
    Parameters:
    spatial_res: voxel size in mm.
    kernel:  dipole kernel matrix, as calculated by the dipole_kernel function.
    
    Output:
    m1, m2: binary masks defining two regions in the Fourier domain, for NDI stopping.
    
    """
    N = kernel.shape

    center = (1 + np.floor_divide(N, 2))

    rin = np.max(spatial_res) * 0.65 / 2  #Radial boundaries of the masks are defined as absolute.
    rout = np.max(spatial_res) * 0.95 / 2  #frequency values [1/mm]. Please modify if needed.

    kx = np.arange(1, N[0] + 1).astype(float)
    ky = np.arange(1, N[1] + 1).astype(float)
    kz = np.arange(1, N[2] + 1).astype(float)

    kx -= center[0]
    ky -= center[1]
    kz -= center[2]

    delta_kx = spatial_res[0,0] / N[0]
    delta_ky = spatial_res[0,1] / N[1]
    delta_kz = spatial_res[0,2] / N[2]

    kx *= delta_kx
    ky *= delta_ky
    kz *= delta_kz

    kx = kx[:, np.newaxis, np.newaxis]
    ky = ky[np.newaxis, :, np.newaxis]
    kz = kz[np.newaxis, np.newaxis, :]

    kx = np.tile(kx, (1, N[1], N[2]))
    ky = np.tile(ky, (N[0], 1, N[2]))
    kz = np.tile(kz, (N[0], N[1], 1))

    k2 = kx ** 2 + ky ** 2 + kz ** 2

    m0 = np.ones(N)
    m0[k2 > rout ** 2] = 0.0
    m0[k2 < rin ** 2] = 0.0
    m0 = np.fft.fftshift(m0)  # Internal mask defining the radial range.

    kernel = kernel.cpu().numpy().astype(np.float32)

    m1 = m0 * (np.logical_xor(np.abs(kernel) > 0.15, np.abs(kernel) > 0.2))  # green
    m2 = m0 * (np.logical_xor(np.abs(kernel) > 0.225, np.abs(kernel) > 0.275))  # cyan

    return m1, m2


def compute_freqe_auto(chi, m1, m2):
    """
    
    Calculates the Mean Amplitude of the energy inside masks
    m1, m2, and (optional) m3 in the Fourier domain of QSM reconstruction chi.

    The estimation of the Mean Amplitude may be the mean magnitude
    of the coefficients (power = 1) or the squared mean of the
    coefficients (power = 2). Both are robust estimators.

    """
    fchi = torch.abs(torch.fft.fftn(chi))

    n1 = torch.sum(m1)
    n2 = torch.sum(m2)
    
    e1 = torch.sum(fchi * m1) / n1
    e2 = torch.sum(fchi * m2) / n2

    return e1, e2