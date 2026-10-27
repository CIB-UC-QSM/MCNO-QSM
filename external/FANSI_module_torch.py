

# import time
# import numpy as np
# import torch
# import torch.fft as fft

# def fansiTV(params):
#     """
    
#     Nonlinear/Linear QSM and Total Variation regularization with spatially variable fidelity
#     and regularization weights.

#     This uses ADMM to solve the functional. 
    
#     Parameters: params - structure with 
#     Required fields:
#     params.input: local field map.
#     params.alpha1: gradient penalty (L1-norm) or regularization weight.
#     Optional fields:
#     params.K: dipole kernel in the frequency space.
#     params.mu1: gradient consistency weight (ADMM weight, recommended = 100*alpha1).
#     params.mu2: fidelity consistency weight (ADMM weight, recommended value = 1.0).
#     params.maxOuterIter: maximum number of iterations (recommended = 150).
#     params.tolUpdate: convergence limit, update ratio of the solution (recommended = 0.1).
#     params.tolDelta: (recommended = 1e-6).
#     params.weight: data fidelity spatially variable weight (recommended = magnitude_data). 
#     params.regWeight: regularization spatially variable weight.
#     params.isNonlinear: Linear or nonlinear algorithm?(default = true).
#     params.voxelSize: voxel size (default value = [1,1,1]).
    
#     Output: out - structure with the following fields:
#     out.x: calculated susceptibility map.
#     out.iter: number of iterations needed.
#     out.time: total elapsed time (including pre-calculations).
    
#     """
    
#     start_time = time.time()
#     # Required parameters
#     Kernel = torch.tensor(params["K"], dtype=torch.float32)
#     phase = torch.tensor(params["input"], dtype=torch.float64)
    
#     alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float)
#     N = phase.shape
#     maxOuterIter = params.get("maxOuterIter", 150)
    
#     weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
#     W = weight * weight    
#     mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

#     tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
#     voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
#     regWeight = torch.from_numpy(np.ones(N + (3,)))
    
#     isNonlinear = params.get("isNonlinear", False)
    
#     if isNonlinear:
#         tolDelta = torch.tensor(params.get("tolDelta", 1E-6))

#     # Redefinition of variable for computational efficiency
#     Wy = (W * phase) / (W + mu2)

#     # Variable initialization
#     z_dx = torch.zeros(N, dtype=torch.float32)
#     z_dy = torch.zeros(N, dtype=torch.float32)
#     z_dz = torch.zeros(N, dtype=torch.float32)

#     s_dx = torch.zeros(N, dtype=torch.float32)
#     s_dy = torch.zeros(N, dtype=torch.float32)
#     s_dz = torch.zeros(N, dtype=torch.float32)


#     x = torch.zeros(N, dtype=torch.float32)

#     z2 = Wy  # start with something similar to the input phase, weighted to reduce noise
    
#     s2 = torch.zeros(N, dtype=torch.float32)

#     alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency

#     # Define the operators
#     k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='xy')

#     E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
#     E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
#     E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]

#     E1t = torch.conj(E1)
#     E2t = torch.conj(E2)
#     E3t = torch.conj(E3)

#     EE2 = E1t * E1 + E2t * E2 + E3t * E3
    
#     zero = torch.tensor(0) 
    
#     torch.cuda.empty_cache()
#     # Move variables to GPU if available  
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     Kernel = Kernel.to(device)
#     phase = phase.to(device)
#     x = x.to(device)
#     alpha1 = alpha1.to(device)
#     Wy = Wy.to(device)
#     W = W.to(device)
#     mu1 = mu1.to(device)
#     mu2 = mu2.to(device)
#     tolUpdate = tolUpdate.to(device)
#     regWeight = regWeight.to(device)
#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)
#     E1t = E1t.to(device)
#     E2t = E2t.to(device)
#     E3t = E3t.to(device)
#     EE2 = EE2.to(device)
#     z_dx = z_dx.to(device)
#     z_dy = z_dy.to(device)
#     z_dz = z_dz.to(device)
#     s_dx = s_dx.to(device)
#     s_dy = s_dy.to(device)
#     s_dz = s_dz.to(device)
#     z2 = z2.to(device)
#     s2 = s2.to(device)
#     zero = zero.to(device)
    
#     Kt = torch.conj(Kernel)
#     regTime = time.time()
    
#     for t in range(maxOuterIter):
#         # Update x: susceptibility estimate
#         tx = E1t * fft.fftn(z_dx - s_dx)
#         ty = E2t * fft.fftn(z_dy - s_dy)
#         tz = E3t * fft.fftn(z_dz - s_dz)

#         x_prev = x.clone()
#         x = torch.real(fft.ifftn((mu1 * (tx + ty + tz) + mu2 * Kt * fft.fftn(z2 - s2)) / (torch.finfo(torch.float64).eps + mu2 * torch.abs(Kernel) ** 2 + mu1 * EE2)))
            
#         del tx
#         del ty
#         del tz 
            
#         x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        
#         print(f'Iter: {t+1}   Update: {x_update:.6f}')
            
#         if x_update < tolUpdate or torch.isnan(x_update):
#             print("\nEarly stopping reached.")
#             break

#         if t < maxOuterIter:
#             # Update z: gradient variable
#             Fx = fft.fftn(x)
#             x_dx = torch.real(fft.ifftn(E1 * Fx))
#             x_dy = torch.real(fft.ifftn(E2 * Fx))
#             x_dz = torch.real(fft.ifftn(E3 * Fx))

#             z_dx = torch.maximum(torch.abs(x_dx + s_dx) - regWeight[:, :, :, 0] * alpha_over_mu, zero) * torch.sign(x_dx + s_dx)
#             z_dy = torch.maximum(torch.abs(x_dy + s_dy) - regWeight[:, :, :, 1] * alpha_over_mu, zero) * torch.sign(x_dy + s_dy)
#             z_dz = torch.maximum(torch.abs(x_dz + s_dz) - regWeight[:, :, :, 2] * alpha_over_mu, zero) * torch.sign(x_dz + s_dz)

#             # Update s: Lagrange multiplier
#             s_dx = s_dx + x_dx - z_dx
#             s_dy = s_dy + x_dy - z_dy
#             s_dz = s_dz + x_dz - z_dz

#             del x_dx
#             del x_dy
#             del x_dz                

#             # Update z2 and s2: data consistency
#             rhs_z2 = mu2 * torch.real(fft.ifftn(Kernel * Fx) + s2)
                
#             if isNonlinear:
#                 z2 = rhs_z2 / mu2
#                 # Newton-Raphson method
#                 delta = float('inf')
#                 inn = 0
#                 while (delta > tolDelta and inn < 10):
#                     inn = inn + 1
#                     norm_old = torch.norm(z2.reshape(-1))
                
#                     update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
                
#                     z2 = z2 - update
#                     delta = torch.norm(update.reshape(-1)) / norm_old
            
#                 del update
#             else:
#                 z2 = Wy + rhs_z2 / (W + mu2)
                
#             del rhs_z2
            
#             s2 = s2 + torch.real(fft.ifftn(Kernel * Fx)) - z2
                    
#             del Fx             
          
#     print('\n')
    
#     # Extract output values
#     end_time = time.time()
#     reg_time = end_time - regTime
#     total_time = end_time - start_time
#     print(f'Regularization Time: {reg_time:.4f} seconds')
#     print(f'Total Execution Time: {total_time:.4f} seconds')

#     out = {"x": x.cpu().detach().numpy().astype(np.float32), "iter": t+1, "time": total_time}
    
#     return out


# def fansiTGV(params):
#     """

#     Nonlinear/Linear QSM and Total Generalized Variation regularization with spatially variable fidelity
#     and regularization weights.

#     This uses ADMM to solve the functional.

#     Parameters: params - structure with
#     Required fields:
#     params.input: local field map.
#     params.alpha1: gradient penalty (L1-norm) or regularization weight.
#     Optional fields:
#     params.K: dipole kernel in the frequency space.
#     params.alpha0: curvature L1 penalty, regularization weight (recommended = 2*alpha1).
#     params.mu1: gradient consistency weight (ADMM weight, recommended = 100*alpha1).
#     params.mu2: fidelity consistency weight (ADMM weight, recommended value = 1.0).
#     params.mu0: curvature consistency weight (ADMM weight, recommended = 2*mu1).
#     params.maxOuterIter: maximum number of iterations (recommended = 150).
#     params.tolUpdate: convergence limit, update ratio of the solution (recommended = 0.1).
#     params.tolDelta: (recommended = 1e-6).
#     params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
#     params.regWeight: regularization spatially variable weight.
#     params.isNonlinear: Linear or nonlinear algorithm?(default = true).
#     params.voxelSize: voxel size (default value = [1,1,1]).

#     Output: out - structure with the following fields:
#     out.x: calculated susceptibility map.
#     out.iter: number of iterations needed.
#     out.time: total elapsed time (including pre-calculations).

#     """
    
#     start_time = time.time()
#     # Required parameters
#     Kernel = torch.tensor(params["K"], dtype=torch.float32)
#     phase = torch.tensor(params["input"], dtype=torch.float64)
    
#     alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float32)
#     alpha0 = params.get("alpha0", 2 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)    
#     N = phase.shape
#     maxOuterIter = params.get("maxOuterIter", 150)
    
#     weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
#     W = weight * weight
#     mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu0 = params.get("mu0", 200 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

#     tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
#     voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
#     regWeight = torch.from_numpy(np.ones(N + (3,)))
    
#     isNonlinear = params.get("isNonlinear", False)
    
    
#     if isNonlinear:
#         tolDelta = torch.tensor(params.get("tolDelta", 1E-6))
#         z2 =  torch.fft.fftn(W * phase / (W + mu2))
#     else:
#         # Redefine variable for computational efficiency
#         FWy = torch.fft.fftn(W * phase / (W + mu2))
#         z2 = FWy  # start with something similar to the input phase, weighted to reduce noise
               
#     alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency
#     alpha0_over_mu0 = (alpha0 / mu0).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency
    
#     # Precompute gradient-related matrices
#     k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='xy')
#     E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
#     E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
#     E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]

#     #z2 = FWy.clone()  # start with something similar to the input phase, weighted to reduce noise
#     s2 = torch.zeros(N)
   
#     # Allocate memory for first order gradient
#     s1_1 = torch.zeros(N)
#     s1_2 = torch.zeros(N)
#     s1_3 = torch.zeros(N)
#     z1_1 = torch.zeros(N)
#     z1_2 = torch.zeros(N)
#     z1_3 = torch.zeros(N)

#     # Allocate memory for symmetrized gradient
#     s0_1 = torch.zeros(N)
#     s0_2 = torch.zeros(N)
#     s0_3 = torch.zeros(N)
#     s0_4 = torch.zeros(N)
#     s0_5 = torch.zeros(N)
#     s0_6 = torch.zeros(N)
#     z0_1 = torch.zeros(N)
#     z0_2 = torch.zeros(N)
#     z0_3 = torch.zeros(N)
#     z0_4 = torch.zeros(N)
#     z0_5 = torch.zeros(N)
#     z0_6 = torch.zeros(N)

#     x = torch.zeros(N)
#     zero = torch.tensor(0)
    
#     # Move variables to GPU if available  
#     torch.cuda.empty_cache()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     if isNonlinear:
#         phase = phase.to(device)
#     else:
#         FWy = FWy.to(device)
    
#     z1_1 = z1_1.to(device)
#     z1_2 = z1_2.to(device)
#     z1_3 = z1_3.to(device)
    
#     z0_1 = z0_1.to(device)
#     z0_2 = z0_2.to(device)
#     z0_3 = z0_3.to(device)
#     z0_4 = z0_4.to(device)
#     z0_5 = z0_5.to(device)
#     z0_6 = z0_6.to(device)
    
#     s1_1 = s1_1.to(device)
#     s1_2 = s1_2.to(device)
#     s1_3 = s1_3.to(device)
    
#     s0_1 = s0_1.to(device)
#     s0_2 = s0_2.to(device)
#     s0_3 = s0_3.to(device)
#     s0_4 = s0_4.to(device)
#     s0_5 = s0_5.to(device)
#     s0_6 = s0_6.to(device)    
    
#     x = x.to(device)
#     Kernel = Kernel.to(device)
    
#     z2 = z2.to(device)
#     s2 = s2.to(device)
    
#     tolUpdate = tolUpdate.to(device)
    
#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)
    
#     regWeight = regWeight.to(device)
#     mu1 = mu1.to(device)
#     mu2 = mu2.to(device)
#     mu0 = mu0.to(device)
#     W = W.to(device)
#     alpha1 = alpha1.to(device)
#     alpha0 = alpha0.to(device)
#     zero = zero.to(device)
    
#     Et1, Et2, Et3 = torch.conj(E1), torch.conj(E2), torch.conj(E3)
#     Kt = torch.conj(Kernel)
    
#     E1tE1 = Et1 * E1
#     E2tE2 = Et2 * E2
#     E3tE3 = Et3 * E3
#     mu0_over_2_E1tE2 = mu0 / 2 * Et1 * E2
#     mu0_over_2_E1tE3 = mu0 / 2 * Et1 * E3
#     mu0_over_2_E2tE3 = mu0 / 2 * Et2 * E3
    
#     KtK_mu1_E_sos = mu2 * Kt * Kernel + mu1 * (E1tE1 + E2tE2 + E3tE3)
#     mu1I_mu0_E_wsos1 = mu1 + mu0 * (E1tE1 + (E2tE2 + E3tE3) / 2)
#     mu1I_mu0_E_wsos2 = mu1 + mu0 * (E1tE1 / 2 + E2tE2 + E3tE3 / 2)
#     mu1I_mu0_E_wsos3 = mu1 + mu0 * ((E1tE1 + E2tE2) / 2 + E3tE3)

#     del E1tE1  
#     del E2tE2
#     del E3tE3
    
#     # Precomputation for Cramer's Rule
#     a1 = KtK_mu1_E_sos
#     a2 = mu1I_mu0_E_wsos1
#     a3 = mu1I_mu0_E_wsos2
#     a4 = mu1I_mu0_E_wsos3
#     a5 = -mu1 * E1
#     a6 = -mu1 * E2
#     a7 = mu0_over_2_E1tE2
#     a8 = -mu1 * E3
#     a9 = mu0_over_2_E1tE3
#     a10 = mu0_over_2_E2tE3
    
#     a5t = torch.conj(a5)
#     a6t = torch.conj(a6)
#     a7t = torch.conj(a7)
#     a8t = torch.conj(a8)
#     a9t = torch.conj(a9)
#     a10t = torch.conj(a10)
    
#     # Free GPU memory for specific variables
#     del KtK_mu1_E_sos
#     del mu1I_mu0_E_wsos1
#     del mu1I_mu0_E_wsos2
#     del mu1I_mu0_E_wsos3
    
#     # For x
#     D11 = a2 * a3 * a4 + a7t * a9 * a10t + a7 * a9t * a10 - a3 * a9 * a9t - a2 * a10 * a10t - a4 * a7 * a7t
#     D21 = a3 * a4 * a5t + a6t * a9 * a10t + a7 * a8t * a10 - a3 * a8t * a9 - a5t * a10 * a10t - a4 * a6t * a7
#     D31 = a4 * a5t * a7t + a6t * a9 * a9t + a2 * a8t * a10 - a7t * a8t * a9 - a5t * a9t * a10 - a2 * a4 * a6t
#     D41 = a5t * a7t * a10t + a6t * a7 * a9t + a2 * a3 * a8t - a7 * a7t * a8t - a3 * a5t * a9t - a2 * a6t * a10t

#     # For vx
#     D12 = a3 * a4 * a5 + a7t * a8 * a10t + a6 * a9t * a10 - a3 * a8 * a9t - a5 * a10 * a10t - a4 * a6 * a7t
#     D22 = a1 * a3 * a4 + a6t * a8 * a10t + a6 * a8t * a10 - a3 * a8 * a8t - a1 * a10 * a10t - a4 * a6 * a6t
#     D32 = a1 * a4 * a7t + a6t * a8 * a9t + a5 * a8t * a10 - a7t * a8 * a8t - a1 * a9t * a10 - a4 * a5 * a6t
#     D42 = a1 * a7t * a10t + a6 * a6t * a9t + a3 * a5 * a8t - a6 * a7t * a8t - a1 * a3 * a9t - a5 * a6t * a10t

#     # For vy
#     D13 = a4 * a5 * a7 + a2 * a8 * a10t + a6 * a9 * a9t - a7 * a8 * a9t - a5 * a9 * a10t - a2 * a4 * a6
#     D23 = a1 * a4 * a7 + a5t * a8 * a10t + a6 * a8t * a9 - a7 * a8 * a8t - a1 * a9 * a10t - a4 * a5t * a6
#     D33 = a1 * a2 * a4 + a5t * a8 * a9t + a5 * a8t * a9 - a2 * a8 * a8t - a1 * a9 * a9t - a4 * a5 * a5t
#     D43 = a1 * a2 * a10t + a5t * a6 * a9t + a5 * a7 * a8t - a2 * a6 * a8t - a1 * a7 * a9t - a5 * a5t * a10t

#     # For vz
#     D14 = a5 * a7 * a10 + a2 * a3 * a8 + a6 * a7t * a9 - a7 * a7t * a8 - a3 * a5 * a9 - a2 * a6 * a10
#     D24 = a1 * a7 * a10 + a3 * a5t * a8 + a6 * a6t * a9 - a6t * a7 * a8 - a1 * a3 * a9 - a5t * a6 * a10
#     D34 = a1 * a2 * a10 + a5t * a7t * a8 + a5 * a6t * a9 - a2 * a6t * a8 - a1 * a7t * a9 - a5 * a5t * a10
#     D44 = a1 * a2 * a3 + a5t * a6 * a7t + a5 * a6t * a7 - a2 * a6 * a6t - a1 * a7 * a7t - a3 * a5 * a5t

#     det_A = a1 * D11 - a5 * D21 + a6 * D31 - a8 * D41
#     det_Ainv = 1 / (torch.finfo(torch.float64).eps + det_A)
    
#     # Free GPU memory for specific variables
#     del mu0_over_2_E1tE2
#     del mu0_over_2_E1tE3
#     del mu0_over_2_E2tE3
#     del a1
#     del a2
#     del a3
#     del a4
#     del a5
#     del a6
#     del a7
#     del a8
#     del a9
#     del a10
#     del a6t
#     del a7t
#     del a8t
#     del a9t
#     del a10t
#     del det_A
    
#     # You can clear additional variables as needed
#     # del D11, D21, D31, D41, D12, D22, D32, D42, D13, D23, D33, D43, D14, D24, D34, D44
    
#     reg_time = time.time()
    
#     for t in range(maxOuterIter):
            
#         # Update x and v: susceptibility estimate
#         F_z0_minus_s0_1 = torch.fft.fftn(z0_1 - s0_1)
#         F_z0_minus_s0_2 = torch.fft.fftn(z0_2 - s0_2)
#         F_z0_minus_s0_3 = torch.fft.fftn(z0_3 - s0_3)
#         F_z0_minus_s0_4 = torch.fft.fftn(z0_4 - s0_4)
#         F_z0_minus_s0_5 = torch.fft.fftn(z0_5 - s0_5)
#         F_z0_minus_s0_6 = torch.fft.fftn(z0_6 - s0_6)
    
#         F_z1_minus_s1_1 = torch.fft.fftn(z1_1 - s1_1)
#         F_z1_minus_s1_2 = torch.fft.fftn(z1_2 - s1_2)
#         F_z1_minus_s1_3 = torch.fft.fftn(z1_3 - s1_3)
        
#         mu2_Kt = mu2 * Kt
#         Et1_F_z1_minus_s1_1 = Et1 * F_z1_minus_s1_1
#         Et2_F_z1_minus_s1_2 = Et2 * F_z1_minus_s1_2
#         Et3_F_z1_minus_s1_3 = Et3 * F_z1_minus_s1_3
        
#         rhs1 = mu2_Kt * (z2 - s2) + mu1 * (Et1_F_z1_minus_s1_1 + Et2_F_z1_minus_s1_2 + Et3_F_z1_minus_s1_3)
#         rhs2 = -mu1 * F_z1_minus_s1_1 + mu0 * (Et1 * F_z0_minus_s0_1 + Et2 * F_z0_minus_s0_4 + Et3 * F_z0_minus_s0_5)
#         rhs3 = -mu1 * F_z1_minus_s1_2 + mu0 * (Et2 * F_z0_minus_s0_2 + Et1 * F_z0_minus_s0_4 + Et3 * F_z0_minus_s0_6)
#         rhs4 = -mu1 * F_z1_minus_s1_3 + mu0 * (Et3 * F_z0_minus_s0_3 + Et1 * F_z0_minus_s0_5 + Et2 * F_z0_minus_s0_6)
            
#         # Free GPU memory for specific variables
#         del F_z0_minus_s0_1
#         del F_z0_minus_s0_2
#         del F_z0_minus_s0_3
#         del F_z0_minus_s0_4
#         del F_z0_minus_s0_5
#         del F_z0_minus_s0_6
#         del F_z1_minus_s1_1
#         del F_z1_minus_s1_2
#         del F_z1_minus_s1_3

#         # Cramer's rul
#         Fx = (rhs1 * D11 - rhs2 * D21 + rhs3 * D31 - rhs4 * D41) * det_Ainv
#         Fv1 = (-rhs1 * D12 + rhs2 * D22 - rhs3 * D32 + rhs4 * D42) * det_Ainv
#         Fv2 = (rhs1 * D13 - rhs2 * D23 + rhs3 * D33 - rhs4 * D43) * det_Ainv
#         Fv3 = (-rhs1 * D14 + rhs2 * D24 - rhs3 * D34 + rhs4 * D44) * det_Ainv
            
#         # Free GPU memory for specific variables
#         del rhs1
#         del rhs2
#         del rhs3
#         del rhs4
            
#         x_prev = x
#         x = torch.real(torch.fft.ifftn(Fx))
#         v1 = torch.real(torch.fft.ifftn(Fv1))
#         v2 = torch.real(torch.fft.ifftn(Fv2))
#         v3 = torch.real(torch.fft.ifftn(Fv3))
    
#         x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
#         print(f'Iter: {t+1}   Update: {x_update:.6f}')
          
#         if x_update < tolUpdate:
#             print("\nEarly stopping reached.")
#             break
        
#         if t < maxOuterIter:
        
#             # Update z: gradient variable
#             Fx = torch.fft.fftn(x)  # Extra step to increase stability from the proximal step
#             Fv1 = torch.fft.fftn(v1)
#             Fv2 = torch.fft.fftn(v2)
#             Fv3 = torch.fft.fftn(v3)
                
#             # Compute gradients for z0 and z1 update
#             Dx1 = torch.real(torch.fft.ifftn(E1 * Fx))
#             Dx2 = torch.real(torch.fft.ifftn(E2 * Fx))
#             Dx3 = torch.real(torch.fft.ifftn(E3 * Fx))
        
#             E_v1 = torch.real(torch.fft.ifftn(E1 * Fv1))
#             E_v2 = torch.real(torch.fft.ifftn(E2 * Fv2))
#             E_v3 = torch.real(torch.fft.ifftn(E3 * Fv3))
#             E_v4 = torch.real(torch.fft.ifftn(E1 * Fv2 + E2 * Fv1)) / 2
#             E_v5 = torch.real(torch.fft.ifftn(E1 * Fv3 + E3 * Fv1)) / 2
#             E_v6 = torch.real(torch.fft.ifftn(E2 * Fv3 + E3 * Fv2)) / 2
            
#             # Free GPU memory for specific variables
#             del Fv1
#             del Fv2
#             del Fv3
            
#             # Update z0: Symm grad
#             z0_1 = torch.max(torch.abs(E_v1 + s0_1) - alpha0_over_mu0, zero) * torch.sign(E_v1 + s0_1)
#             z0_2 = torch.max(torch.abs(E_v2 + s0_2) - alpha0_over_mu0, zero) * torch.sign(E_v2 + s0_2)
#             z0_3 = torch.max(torch.abs(E_v3 + s0_3) - alpha0_over_mu0, zero) * torch.sign(E_v3 + s0_3)
#             z0_4 = torch.max(torch.abs(E_v4 + s0_4) - alpha0_over_mu0, zero) * torch.sign(E_v4 + s0_4)
#             z0_5 = torch.max(torch.abs(E_v5 + s0_5) - alpha0_over_mu0, zero) * torch.sign(E_v5 + s0_5)
#             z0_6 = torch.max(torch.abs(E_v6 + s0_6) - alpha0_over_mu0, zero) * torch.sign(E_v6 + s0_6)
                
#             # Update z1: Grad
#             z1_1 = torch.max(torch.abs(Dx1 - v1 + s1_1) - regWeight[..., 0] * alpha_over_mu, zero) * torch.sign(Dx1 - v1 + s1_1)
#             z1_2 = torch.max(torch.abs(Dx2 - v2 + s1_2) - regWeight[..., 1] * alpha_over_mu, zero) * torch.sign(Dx2 - v2 + s1_2)
#             z1_3 = torch.max(torch.abs(Dx3 - v3 + s1_3) - regWeight[..., 2] * alpha_over_mu, zero) * torch.sign(Dx3 - v3 + s1_3)
            
#             # Update z2 and s2: data consistency
#             rhs_z2 = mu2 * torch.real(fft.ifftn(Kernel * Fx + s2))
#             if isNonlinear:
#                 # Update z2 and s2: data consistency
#                 z2 = rhs_z2 / mu2
                    
#                 # Newton-Raphson method
#                 delta = float('inf')
#                 inn = 0
#                 while (delta > tolDelta and inn < 50):
#                     inn = inn + 1
#                     norm_old = torch.norm(z2.reshape(-1))
                
#                     update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
            
#                     z2 -= update
#                     delta = torch.norm(update.reshape(-1)) / norm_old
                    
#                 z2 = fft.fftn(z2)
                
#                 # Free GPU memory for specific variables
#                 del rhs_z2
#                 del update
#             else:
#                 z2 = FWy + torch.fft.fftn(rhs_z2 / (W + mu2))
            
#             s2 = s2 + Kernel * Fx - z2
                    
#             # Update s0 and s1
#             s0_1 = s0_1 + E_v1 - z0_1
#             s0_2 = s0_2 + E_v2 - z0_2
#             s0_3 = s0_3 + E_v3 - z0_3
#             s0_4 = s0_4 + E_v4 - z0_4
#             s0_5 = s0_5 + E_v5 - z0_5
#             s0_6 = s0_6 + E_v6 - z0_6
        
#             s1_1 = s1_1 + Dx1 - v1 - z1_1
#             s1_2 = s1_2 + Dx2 - v2 - z1_2
#             s1_3 = s1_3 + Dx3 - v3 - z1_3
            
#             # Free GPU memory for specific variables
#             del Dx1
#             del Dx2
#             del Dx3
#             del E_v1
#             del E_v2
#             del E_v3
#             del E_v4
#             del E_v5
#             del E_v6
            
#             #s2 = s2 + Kernel * Fx - z2
            
#     # Extract output values
#     end_time = time.time()
#     reg_time = end_time - reg_time
#     total_time = end_time - start_time
#     print(f'Regularization Time: {reg_time:.4f} seconds')
#     print(f'Total Execution Time: {total_time:.4f} seconds')
    
#     out = {"x": x.cpu().detach().numpy().astype(np.float32), "iter": t+1, "time": total_time}
    
#     return out

# import time
# import numpy as np
# import torch
# import torch.fft as fft

# def fansiTV(params):
#     """
    
#     Nonlinear/Linear QSM and Total Variation regularization...
    
#     ... (docstring) ...
#     params.weight: data fidelity spatially variable weight (e.g., magnitude)
#     params.mask: (optional) binary mask for NRMSE calculation. 
#                  If not provided, params["weight"] will be used.
#     params.chi_cosmos: (optional) ground truth susceptibility map (NumPy array) 
#                        to calculate NRMSE at each iteration.
    
#     Output: out - structure with the following fields:
#     ...
#     out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
    
#     """
    
#     start_time = time.time()
#     # Define device FIRST
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Required parameters
#     Kernel = torch.tensor(params["K"], dtype=torch.float32)
#     phase = torch.tensor(params["input"], dtype=torch.float64)
    
#     alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float)
#     N = phase.shape
#     maxOuterIter = params.get("maxOuterIter", 150)

#     # Load fidelity weight (e.g., magnitude)
#     weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
#     W = weight * weight # 'W' is the squared fidelity weight (w*w)
    
#     # Load the separate mask for NRMSE calculation
#     mask_np = params.get("mask", None)
#     if mask_np is None:
#         print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
#         mask_for_rmse = weight.to(device) # Fallback to using fidelity weight
#     else:
#         mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32, device=device)

#     mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

#     tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
#     voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
#     regWeight = torch.from_numpy(np.ones(N + (3,)))
    
#     isNonlinear = params.get("isNonlinear", False)
    
#     if isNonlinear:
#         tolDelta = torch.tensor(params.get("tolDelta", 1E-6))

#     # Redefinition of variable for computational efficiency
#     Wy = (W * phase) / (W + mu2)

#     # Variable initialization
#     z_dx = torch.zeros(N, dtype=torch.float32)
#     z_dy = torch.zeros(N, dtype=torch.float32)
#     z_dz = torch.zeros(N, dtype=torch.float32)

#     s_dx = torch.zeros(N, dtype=torch.float32)
#     s_dy = torch.zeros(N, dtype=torch.float32)
#     s_dz = torch.zeros(N, dtype=torch.float32)

#     x = torch.zeros(N, dtype=torch.float32)
#     z2 = Wy  # start with something similar to the input phase, weighted to reduce noise
#     s2 = torch.zeros(N, dtype=torch.float32)

#     alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency

#     # Define the operators
#     k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')

#     #E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
#     #E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
#     #E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]

#     E1 = 1 - torch.exp(2j * torch.pi * k1 / N[0])
#     E2 = 1 - torch.exp(2j * torch.pi * k2 / N[1])
#     E3 = 1 - torch.exp(2j * torch.pi * k3 / N[2])

#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)

#     E1t = torch.conj(E1)
#     E2t = torch.conj(E2)
#     E3t = torch.conj(E3)

#     EE2 = E1t * E1 + E2t * E2 + E3t * E3
    
#     zero = torch.tensor(0) 
    
#     torch.cuda.empty_cache()
#     # Move variables to GPU
#     Kernel = Kernel.to(device)
#     phase = phase.to(device)
#     x = x.to(device)
#     alpha1 = alpha1.to(device)
#     Wy = Wy.to(device)
#     W = W.to(device)
#     weight = weight.to(device) 
#     mu1 = mu1.to(device)
#     mu2 = mu2.to(device)
#     tolUpdate = tolUpdate.to(device)
#     regWeight = regWeight.to(device)
#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)
#     E1t = E1t.to(device)
#     E2t = E2t.to(device)
#     E3t = E3t.to(device)
#     EE2 = EE2.to(device)
#     z_dx = z_dx.to(device)
#     z_dy = z_dy.to(device)
#     z_dz = z_dz.to(device)
#     s_dx = s_dx.to(device)
#     s_dy = s_dy.to(device)
#     s_dz = s_dz.to(device)
#     z2 = z2.to(device)
#     s2 = s2.to(device)
#     zero = zero.to(device)
    
#     # --- NRMSE History Initialization ---
#     chi_cosmos_gt = params.get("chi_cosmos", None)
#     rmse_history = []
#     chi_cosmos_tensor = None
#     if chi_cosmos_gt is not None:
#         chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
#         # mask_for_rmse is already defined and on device
#         masked_true = mask_for_rmse * chi_cosmos_tensor
#         norm_true = torch.norm(masked_true) + 1e-8 # Add eps for stability
#     # --- End NRMSE Initialization ---
    
#     Kt = torch.conj(Kernel)
#     regTime = time.time()
    
#     for t in range(maxOuterIter):
#         # Update x: susceptibility estimate
#         tx = E1t * fft.fftn(z_dx - s_dx)
#         ty = E2t * fft.fftn(z_dy - s_dy)
#         tz = E3t * fft.fftn(z_dz - s_dz)

#         x_prev = x.clone()
#         x = torch.real(fft.ifftn((mu1 * (tx + ty + tz) + mu2 * Kt * fft.fftn(z2 - s2)) / (torch.finfo(torch.float64).eps + mu2 * torch.abs(Kernel) ** 2 + mu1 * EE2)))
            
#         del tx
#         del ty
#         del tz 
            
#         # --- NRMSE Calculation ---
#         if chi_cosmos_tensor is not None:
#             # Calculate masked error norm
#             masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
#             norm_error = torch.norm(masked_error)
            
#             # Calculate NRMSE percentage
#             nrmse = 100 * norm_error / norm_true
#             rmse_history.append(nrmse.item())
#         # --- End NRMSE Calculation ---
            
#         x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        
#         if chi_cosmos_tensor is not None:
#             print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
#         else:
#             print(f'Iter: {t+1}   Update: {x_update:.6f}')
            
#         if x_update < tolUpdate or torch.isnan(x_update):
#             print("\nEarly stopping reached.")
#             break

#         if t < maxOuterIter:
#             # Update z: gradient variable
#             Fx = fft.fftn(x)
#             x_dx = torch.real(fft.ifftn(E1 * Fx))
#             x_dy = torch.real(fft.ifftn(E2 * Fx))
#             x_dz = torch.real(fft.ifftn(E3 * Fx))

#             z_dx = torch.maximum(torch.abs(x_dx + s_dx) - regWeight[:, :, :, 0] * alpha_over_mu, zero) * torch.sign(x_dx + s_dx)
#             z_dy = torch.maximum(torch.abs(x_dy + s_dy) - regWeight[:, :, :, 1] * alpha_over_mu, zero) * torch.sign(x_dy + s_dy)
#             z_dz = torch.maximum(torch.abs(x_dz + s_dz) - regWeight[:, :, :, 2] * alpha_over_mu, zero) * torch.sign(x_dz + s_dz)

#             # Update s: Lagrange multiplier
#             s_dx = s_dx + x_dx - z_dx
#             s_dy = s_dy + x_dy - z_dy
#             s_dz = s_dz + x_dz - z_dz

#             del x_dx
#             del x_dy
#             del x_dz              

#             # Update z2 and s2: data consistency
#             rhs_z2 = mu2 * torch.real(fft.ifftn(Kernel * Fx) + s2)
                
#             if isNonlinear:
#                 z2 = rhs_z2 / mu2
#                 # Newton-Raphson method
#                 delta = float('inf')
#                 inn = 0
#                 while (delta > tolDelta and inn < 10):
#                     inn = inn + 1
#                     norm_old = torch.norm(z2.reshape(-1))
                
#                     update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
                
#                     z2 = z2 - update
#                     delta = torch.norm(update.reshape(-1)) / norm_old
            
#                 del update
#             else:
#                 z2 = Wy + rhs_z2 / (W + mu2)
                
#             del rhs_z2
            
#             s2 = s2 + torch.real(fft.ifftn(Kernel * Fx)) - z2
                    
#             del Fx            
        
#     print('\n')
    
#     # Extract output values
#     end_time = time.time()
#     reg_time = end_time - regTime
#     total_time = end_time - start_time
#     print(f'Regularization Time: {reg_time:.4f} seconds')
#     print(f'Total Execution Time: {total_time:.4f} seconds')

#     out = {
#         "x": x.cpu().detach().numpy().astype(np.float32), 
#         "iter": t+1, 
#         "time": total_time
#     }
    
#     if chi_cosmos_tensor is not None:
#         out["rmse_history"] = rmse_history
    
#     return out

import time
import torch
import torch.fft as fft
import numpy as np

# def fansiTV(params):
#     """
    
#     Nonlinear/Linear QSM and Total Variation regularization...
    
#     ... (docstring) ...
#     params.weight: data fidelity spatially variable weight (e.g., magnitude)
#     params.mask: (optional) binary mask for NRMSE calculation. 
#                  If not provided, params["weight"] will be used.
#     params.chi_cosmos: (optional) ground truth susceptibility map (NumPy array) 
#                         to calculate NRMSE at each iteration.
#     params.isLcurve: (optional) boolean to compute residual and TV norms for L-curve analysis.
    
#     Output: out - structure with the following fields:
#     ...
#     out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
#     out.res_norm: (if isLcurve is True) residual norm of the final solution.
#     out.tv_norm: (if isLcurve is True) Total Variation norm of the final solution.
    
#     """
    
#     start_time = time.time()
#     # Define device FIRST
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Required parameters
#     Kernel = torch.tensor(params["K"], dtype=torch.float32)
#     phase = torch.tensor(params["input"], dtype=torch.float64)
    
#     alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float)
#     N = phase.shape
#     maxOuterIter = params.get("maxOuterIter", 150)

#     # Load fidelity weight (e.g., magnitude)
#     weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
#     W = weight * weight # 'W' is the squared fidelity weight (w*w)
    
#     # Load the separate mask for NRMSE calculation
#     mask_np = params.get("mask", None)
#     if mask_np is None:
#         print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
#         mask_for_rmse = weight.to(device) # Fallback to using fidelity weight
#     else:
#         mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32, device=device)

#     mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

#     tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
#     voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
#     regWeight = torch.from_numpy(np.ones(N + (3,)))
    
#     isNonlinear = params.get("isNonlinear", False)
    
#     # --- L-curve toggle added here ---
#     is_lcurve = params.get("isLcurve", False)

#     if isNonlinear:
#         tolDelta = torch.tensor(params.get("tolDelta", 1E-6))

#     # Redefinition of variable for computational efficiency
#     Wy = (W * phase) / (W + mu2)

#     # Variable initialization
#     z_dx = torch.zeros(N, dtype=torch.float32)
#     z_dy = torch.zeros(N, dtype=torch.float32)
#     z_dz = torch.zeros(N, dtype=torch.float32)

#     s_dx = torch.zeros(N, dtype=torch.float32)
#     s_dy = torch.zeros(N, dtype=torch.float32)
#     s_dz = torch.zeros(N, dtype=torch.float32)

#     x = torch.zeros(N, dtype=torch.float32)
#     z2 = Wy  # start with something similar to the input phase, weighted to reduce noise
#     s2 = torch.zeros(N, dtype=torch.float32)

#     alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency

#     # Define the operators
#     k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')

#     #E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
#     #E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
#     #E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]

#     E1 = 1 - torch.exp(2j * torch.pi * k1 / N[0])
#     E2 = 1 - torch.exp(2j * torch.pi * k2 / N[1])
#     E3 = 1 - torch.exp(2j * torch.pi * k3 / N[2])

#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)

#     E1t = torch.conj(E1)
#     E2t = torch.conj(E2)
#     E3t = torch.conj(E3)

#     EE2 = E1t * E1 + E2t * E2 + E3t * E3
    
#     zero = torch.tensor(0) 
    
#     torch.cuda.empty_cache()
#     # Move variables to GPU
#     Kernel = Kernel.to(device)
#     phase = phase.to(device)
#     x = x.to(device)
#     alpha1 = alpha1.to(device)
#     Wy = Wy.to(device)
#     W = W.to(device)
#     weight = weight.to(device) 
#     mu1 = mu1.to(device)
#     mu2 = mu2.to(device)
#     tolUpdate = tolUpdate.to(device)
#     regWeight = regWeight.to(device)
#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)
#     E1t = E1t.to(device)
#     E2t = E2t.to(device)
#     E3t = E3t.to(device)
#     EE2 = EE2.to(device)
#     z_dx = z_dx.to(device)
#     z_dy = z_dy.to(device)
#     z_dz = z_dz.to(device)
#     s_dx = s_dx.to(device)
#     s_dy = s_dy.to(device)
#     s_dz = s_dz.to(device)
#     z2 = z2.to(device)
#     s2 = s2.to(device)
#     zero = zero.to(device)
    
#     # --- NRMSE History Initialization ---
#     chi_cosmos_gt = params.get("chi_cosmos", None)
#     rmse_history = []
#     chi_cosmos_tensor = None
#     if chi_cosmos_gt is not None:
#         chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
#         # mask_for_rmse is already defined and on device
#         masked_true = mask_for_rmse * chi_cosmos_tensor
#         norm_true = torch.norm(masked_true) + 1e-8 # Add eps for stability
#     # --- End NRMSE Initialization ---
    
#     Kt = torch.conj(Kernel)
#     regTime = time.time()
    
#     for t in range(maxOuterIter):
#         # Update x: susceptibility estimate
#         tx = E1t * fft.fftn(z_dx - s_dx)
#         ty = E2t * fft.fftn(z_dy - s_dy)
#         tz = E3t * fft.fftn(z_dz - s_dz)

#         x_prev = x.clone()
#         x = torch.real(fft.ifftn((mu1 * (tx + ty + tz) + mu2 * Kt * fft.fftn(z2 - s2)) / (torch.finfo(torch.float64).eps + mu2 * torch.abs(Kernel) ** 2 + mu1 * EE2)))
            
#         del tx
#         del ty
#         del tz 
            
#         # --- NRMSE Calculation ---
#         if chi_cosmos_tensor is not None:
#             # Calculate masked error norm
#             masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
#             norm_error = torch.norm(masked_error)
            
#             # Calculate NRMSE percentage
#             nrmse = 100 * norm_error / norm_true
#             rmse_history.append(nrmse.item())
#         # --- End NRMSE Calculation ---
            
#         x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        
#         if chi_cosmos_tensor is not None:
#             print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
#         else:
#             print(f'Iter: {t+1}   Update: {x_update:.6f}')
            
#         if x_update < tolUpdate or torch.isnan(x_update):
#             print("\nEarly stopping reached.")
#             break

#         if t < maxOuterIter:
#             # Update z: gradient variable
#             Fx = fft.fftn(x)
#             x_dx = torch.real(fft.ifftn(E1 * Fx))
#             x_dy = torch.real(fft.ifftn(E2 * Fx))
#             x_dz = torch.real(fft.ifftn(E3 * Fx))

#             z_dx = torch.maximum(torch.abs(x_dx + s_dx) - regWeight[:, :, :, 0] * alpha_over_mu, zero) * torch.sign(x_dx + s_dx)
#             z_dy = torch.maximum(torch.abs(x_dy + s_dy) - regWeight[:, :, :, 1] * alpha_over_mu, zero) * torch.sign(x_dy + s_dy)
#             z_dz = torch.maximum(torch.abs(x_dz + s_dz) - regWeight[:, :, :, 2] * alpha_over_mu, zero) * torch.sign(x_dz + s_dz)

#             # Update s: Lagrange multiplier
#             s_dx = s_dx + x_dx - z_dx
#             s_dy = s_dy + x_dy - z_dy
#             s_dz = s_dz + x_dz - z_dz

#             del x_dx
#             del x_dy
#             del x_dz              

#             # Update z2 and s2: data consistency
#             rhs_z2 = mu2 * torch.real(fft.ifftn(Kernel * Fx) + s2)
                
#             if isNonlinear:
#                 z2 = rhs_z2 / mu2
#                 # Newton-Raphson method
#                 delta = float('inf')
#                 inn = 0
#                 while (delta > tolDelta and inn < 10):
#                     inn = inn + 1
#                     norm_old = torch.norm(z2.reshape(-1))
                
#                     update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
                
#                     z2 = z2 - update
#                     delta = torch.norm(update.reshape(-1)) / norm_old
            
#                 del update
#             else:
#                 z2 = Wy + rhs_z2 / (W + mu2)
                
#             del rhs_z2
            
#             s2 = s2 + torch.real(fft.ifftn(Kernel * Fx)) - z2
                    
#             del Fx            
        
#     print('\n')
    
#     # --- L-curve Norm Calculations Added Here ---
#     if is_lcurve:
#         Fx = fft.fftn(x)
#         Kx = torch.real(fft.ifftn(Kernel * Fx))
        
#         if isNonlinear:
#             res_norm = torch.sum(W * torch.abs(torch.exp(1j * Kx) - torch.exp(1j * phase))**2)
#         else:
#             res_norm = torch.sum(W * (Kx - phase)**2)

#         # TV norm in Fourier formulation using existing gradient operators
#         x_dx = torch.real(fft.ifftn(E1 * Fx))
#         x_dy = torch.real(fft.ifftn(E2 * Fx))
#         x_dz = torch.real(fft.ifftn(E3 * Fx))
#         tv_norm = torch.sum(torch.abs(x_dx)) + torch.sum(torch.abs(x_dy)) + torch.sum(torch.abs(x_dz))
#     # --- End L-curve Calculations ---
    
#     # Extract output values
#     end_time = time.time()
#     reg_time = end_time - regTime
#     total_time = end_time - start_time
#     print(f'Regularization Time: {reg_time:.4f} seconds')
#     print(f'Total Execution Time: {total_time:.4f} seconds')

#     out = {
#         "x": x.cpu().detach().numpy().astype(np.float32), 
#         "iter": t+1, 
#         "time": total_time
#     }
    
#     if chi_cosmos_tensor is not None:
#         out["rmse_history"] = rmse_history
        
#     # Add L-curve outputs to dictionary if requested
#     if is_lcurve:
#         out["res_norm"] = res_norm.item()
#         out["tv_norm"] = tv_norm.item()
    
#     return out

import time
import torch
import torch.fft as fft
import numpy as np
import torch.nn.functional as F



def _gaussian_kernel1d(sigma, radius, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device='cuda', dtype=torch.float32):
    """Built once, outside the loop -- sw/sigma are fixed across iterations."""
    return [(_gaussian_kernel1d(sigma, r, device, dtype), r) for r in sw]


def _separable_gaussian_filter3d(x, kernels_radii, padding_mode):
    out = x
    for axis, (kernel, radius) in zip((2, 3, 4), kernels_radii):
        shape = [1, 1, 1, 1, 1]
        shape[axis] = kernel.numel()
        k = kernel.view(*shape)
        pad = [0, 0, 0, 0, 0, 0]
        pos = (4 - axis) * 2
        pad[pos] = radius
        pad[pos + 1] = radius
        out = F.pad(out, pad, mode=padding_mode)
        out = F.conv3d(out, k)
    return out


def compute_xsim_torch(img1, img2, mask_bool, kernels_radii, padding_mode='replicate'):
    """
    Scalar-only GPU XSIM. Returns mssim as a GPU tensor (no .item()/.cpu(),
    no full ssim_map materialization/transfer) so it's cheap to call every
    outer iteration of a solver.

    Args:
        img1, img2: torch.Tensor, shape (D,H,W), already on the target device.
        mask_bool: precomputed bool tensor on device, shape (D,H,W).
        kernels_radii: output of _build_xsim_kernels(), precomputed once.
        padding_mode: 'replicate' (default, matches MATLAB imgaussfilt3) or 'reflect'.

    Returns:
        mssim (torch.Tensor, scalar, on device)
    """
    dtype = kernels_radii[0][0].dtype
    img1_t = img1.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    img2_t = img2.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    K1, K2, L = 0.01, 0.001, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    mu1 = _separable_gaussian_filter3d(img1_t, kernels_radii, padding_mode)
    mu2 = _separable_gaussian_filter3d(img2_t, kernels_radii, padding_mode)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    del mu1, mu2

    img1_sq, img2_sq, img1_img2 = img1_t.pow(2), img2_t.pow(2), img1_t * img2_t
    del img1_t, img2_t

    sigma1_sq = _separable_gaussian_filter3d(img1_sq, kernels_radii, padding_mode) - mu1_sq
    del img1_sq
    sigma2_sq = _separable_gaussian_filter3d(img2_sq, kernels_radii, padding_mode) - mu2_sq
    del img2_sq
    sigma12 = _separable_gaussian_filter3d(img1_img2, kernels_radii, padding_mode) - mu1_mu2
    del img1_img2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    del sigma12
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    del mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq

    ssim_map = (numerator / denominator).squeeze(0).squeeze(0)
    del numerator, denominator

    mssim = ssim_map[mask_bool].mean()   # stays on GPU, no sync
    del ssim_map
    return mssim

def fansiTV(params):
    """
    
    Nonlinear/Linear QSM and Total Variation regularization...
    
    ... (docstring) ...
    params.weight: data fidelity spatially variable weight (e.g., magnitude)
    params.mask: (optional) binary mask for NRMSE calculation. 
                 If not provided, params["weight"] will be used.
    params.chi_cosmos: (optional) ground truth susceptibility map (NumPy array) 
                        to calculate NRMSE at each iteration.
    params.isLcurve: (optional) boolean to compute residual and TV norms for L-curve analysis.
    params.isXSIM: (optional) boolean to compute XSIM at each iteration (default: False).
    params.scale: (optional) scale factor to bring x / chi_cosmos into the ~1 ppm
                  range expected by the XSIM constants, before computing XSIM.
    
    Output: out - structure with the following fields:
    ...
    out.rmse_history: (if chi_cosmos is provided) list of NRMSE values (in %) per iteration.
    out.xsim_history: (if chi_cosmos is provided and isXSIM) list of XSIM values per iteration.
    out.res_norm: (if isLcurve is True) residual norm of the final solution.
    out.tv_norm: (if isLcurve is True) Total Variation norm of the final solution.
    
    """
    
    start_time = time.time()
    # Define device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Required parameters
    Kernel = torch.tensor(params["K"], dtype=torch.float32)
    phase = torch.tensor(params["input"], dtype=torch.float64)
    
    alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float)
    N = phase.shape
    maxOuterIter = params.get("maxOuterIter", 150)

    # Load fidelity weight (e.g., magnitude)
    weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
    W = weight * weight # 'W' is the squared fidelity weight (w*w)
    
    # Load the separate mask for NRMSE calculation
    mask_np = params.get("mask", None)
    if mask_np is None:
        print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
        mask_for_rmse = weight.to(device) # Fallback to using fidelity weight
    else:
        mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32, device=device)

    mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
    mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

    tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
    voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
    regWeight = torch.from_numpy(np.ones(N + (3,)))
    
    isNonlinear = params.get("isNonlinear", False)
    
    # --- L-curve toggle added here ---
    is_lcurve = params.get("isLcurve", False)

    # --- XSIM toggle added here ---
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("scale", 1.0), dtype=torch.float32).to(device)

    if isNonlinear:
        tolDelta = torch.tensor(params.get("tolDelta", 1E-6), device=device)

    # Redefinition of variable for computational efficiency
    Wy = (W * phase) / (W + mu2)

    # Variable initialization
    z_dx = torch.zeros(N, dtype=torch.float32)
    z_dy = torch.zeros(N, dtype=torch.float32)
    z_dz = torch.zeros(N, dtype=torch.float32)

    s_dx = torch.zeros(N, dtype=torch.float32)
    s_dy = torch.zeros(N, dtype=torch.float32)
    s_dz = torch.zeros(N, dtype=torch.float32)

    x = torch.zeros(N, dtype=torch.float32)
    z2 = Wy  # start with something similar to the input phase, weighted to reduce noise
    s2 = torch.zeros(N, dtype=torch.float32)

    alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency

    # Define the operators
    k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')

    #E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
    #E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
    #E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]

    E1 = 1 - torch.exp(2j * torch.pi * k1 / N[0])
    E2 = 1 - torch.exp(2j * torch.pi * k2 / N[1])
    E3 = 1 - torch.exp(2j * torch.pi * k3 / N[2])

    E1 = E1.to(device)
    E2 = E2.to(device)
    E3 = E3.to(device)

    E1t = torch.conj(E1)
    E2t = torch.conj(E2)
    E3t = torch.conj(E3)

    EE2 = E1t * E1 + E2t * E2 + E3t * E3
    
    zero = torch.tensor(0) 
    
    torch.cuda.empty_cache()
    # Move variables to GPU
    Kernel = Kernel.to(device)
    phase = phase.to(device)
    x = x.to(device)
    alpha1 = alpha1.to(device)
    Wy = Wy.to(device)
    W = W.to(device)
    weight = weight.to(device) 
    mu1 = mu1.to(device)
    mu2 = mu2.to(device)
    tolUpdate = tolUpdate.to(device)
    regWeight = regWeight.to(device)
    E1 = E1.to(device)
    E2 = E2.to(device)
    E3 = E3.to(device)
    E1t = E1t.to(device)
    E2t = E2t.to(device)
    E3t = E3t.to(device)
    EE2 = EE2.to(device)
    z_dx = z_dx.to(device)
    z_dy = z_dy.to(device)
    z_dz = z_dz.to(device)
    s_dx = s_dx.to(device)
    s_dy = s_dy.to(device)
    s_dz = s_dz.to(device)
    z2 = z2.to(device)
    s2 = s2.to(device)
    zero = zero.to(device)
    
    # --- NRMSE History Initialization ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
        # mask_for_rmse is already defined and on device
        masked_true = mask_for_rmse * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8 # Add eps for stability
    # --- End NRMSE Initialization ---

    # --- XSIM History Initialization ---
    # Preallocated GPU buffer + precomputed kernels/mask -- avoids rebuilding
    # the Gaussian kernels and casting the mask every iteration, and avoids
    # a host sync on every append (only synced once, after the loop).
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_buffer = torch.zeros(maxOuterIter, device=device, dtype=torch.float32)
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask_for_rmse.bool()
    # --- End XSIM Initialization ---
    
    Kt = torch.conj(Kernel)
    regTime = time.time()
    update_history = []
    for t in range(maxOuterIter):
        # Update x: susceptibility estimate
        tx = E1t * fft.fftn(z_dx - s_dx)
        ty = E2t * fft.fftn(z_dy - s_dy)
        tz = E3t * fft.fftn(z_dz - s_dz)

        x_prev = x.clone()
        x = torch.real(fft.ifftn((mu1 * (tx + ty + tz) + mu2 * Kt * fft.fftn(z2 - s2)) / (torch.finfo(torch.float64).eps + mu2 * torch.abs(Kernel) ** 2 + mu1 * EE2)))
            
        del tx
        del ty
        del tz 
            
        # --- NRMSE Calculation ---
        if chi_cosmos_tensor is not None:
            # Calculate masked error norm
            masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            
            # Calculate NRMSE percentage
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # --- XSIM Calculation ---
            if isXSIM:
                mssim = compute_xsim_torch( (x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_buffer[t] = mssim
            # --- End XSIM Calculation ---
        # --- End NRMSE Calculation ---
            
        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())
        
        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_buffer[t]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')
            
        if x_update < tolUpdate or torch.isnan(x_update):
            print("\nEarly stopping reached.")
            break

        if t < maxOuterIter:
            # Update z: gradient variable
            Fx = fft.fftn(x)
            x_dx = torch.real(fft.ifftn(E1 * Fx))
            x_dy = torch.real(fft.ifftn(E2 * Fx))
            x_dz = torch.real(fft.ifftn(E3 * Fx))

            z_dx = torch.maximum(torch.abs(x_dx + s_dx) - regWeight[:, :, :, 0] * alpha_over_mu, zero) * torch.sign(x_dx + s_dx)
            z_dy = torch.maximum(torch.abs(x_dy + s_dy) - regWeight[:, :, :, 1] * alpha_over_mu, zero) * torch.sign(x_dy + s_dy)
            z_dz = torch.maximum(torch.abs(x_dz + s_dz) - regWeight[:, :, :, 2] * alpha_over_mu, zero) * torch.sign(x_dz + s_dz)

            # Update s: Lagrange multiplier
            s_dx = s_dx + x_dx - z_dx
            s_dy = s_dy + x_dy - z_dy
            s_dz = s_dz + x_dz - z_dz

            del x_dx
            del x_dy
            del x_dz              

            # Update z2 and s2: data consistency
            rhs_z2 = mu2 * torch.real(fft.ifftn(Kernel * Fx) + s2)
                
            if isNonlinear:
                z2 = rhs_z2 / mu2
                # Newton-Raphson method
                delta = float('inf')
                inn = 0
                while (delta > tolDelta and inn < 10):
                    inn = inn + 1
                    norm_old = torch.norm(z2.reshape(-1))
                
                    update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
                
                    z2 = z2 - update
                    delta = torch.norm(update.reshape(-1)) / norm_old
            
                del update
            else:
                z2 = Wy + rhs_z2 / (W + mu2)
                
            del rhs_z2
            
            s2 = s2 + torch.real(fft.ifftn(Kernel * Fx)) - z2
                    
            del Fx            
        
    print('\n')
    
    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        Fx = fft.fftn(x)
        Kx = torch.real(fft.ifftn(Kernel * Fx))
        
        if isNonlinear:
            res_norm = torch.sum(W * torch.abs(torch.exp(1j * Kx) - torch.exp(1j * phase))**2)
        else:
            res_norm = torch.sum(W * (Kx - phase)**2)

        # TV norm in Fourier formulation using existing gradient operators
        x_dx = torch.real(fft.ifftn(E1 * Fx))
        x_dy = torch.real(fft.ifftn(E2 * Fx))
        x_dz = torch.real(fft.ifftn(E3 * Fx))
        tv_norm = torch.sum(torch.abs(x_dx)) + torch.sum(torch.abs(x_dy)) + torch.sum(torch.abs(x_dz))
    # --- End L-curve Calculations ---
    
    # Extract output values
    end_time = time.time()
    reg_time = end_time - regTime
    total_time = end_time - start_time
    print(f'Regularization Time: {reg_time:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')

    out = {
        "x": x.cpu().detach().numpy().astype(np.float32), 
        "iter": t+1, 
        "time": total_time,
        "update_history": update_history
    }
    
    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_buffer[:t + 1].cpu().tolist()
        
    # Add L-curve outputs to dictionary if requested
    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["tv_norm"] = tv_norm.item()
    
    return out



# def fansiTGV(params):
#     """
#     ... (docstring) ...
#     """
    
#     start_time = time.time()
#     # Define device FIRST
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Required parameters
#     Kernel = torch.tensor(params["K"], dtype=torch.float32)
#     phase = torch.tensor(params["input"], dtype=torch.float64)
    
#     alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float32)
#     alpha0 = params.get("alpha0", 2 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)      
#     N = phase.shape
#     maxOuterIter = params.get("maxOuterIter", 150)

#     # Load fidelity weight (e.g., magnitude)
#     weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
#     W = weight * weight
    
#     # Load the separate mask for NRMSE calculation
#     mask_np = params.get("mask", None)
#     if mask_np is None:
#         print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
#         mask_for_rmse = weight.to(device) # Fallback to using fidelity weight
#     else:
#         mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32, device=device)

#     mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu0 = params.get("mu0", 200 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
#     mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

#     tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
#     voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
#     regWeight = torch.from_numpy(np.ones(N + (3,)))
    
#     isNonlinear = params.get("isNonlinear", False)
    
    
#     if isNonlinear:
#         tolDelta = torch.tensor(params.get("tolDelta", 1E-6))
#         z2 =  torch.fft.fftn(W * phase / (W + mu2))
#     else:
#         # Redefine variable for computational efficiency
#         FWy = torch.fft.fftn(W * phase / (W + mu2))
#         z2 = FWy  # start with something similar to the input phase, weighted to reduce noise
            
#     alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency
#     alpha0_over_mu0 = (alpha0 / mu0).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency
    
#     # Precompute gradient-related matrices
#     k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')
#     #E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
#     #E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
#     #E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]
#     E1 = 1 - torch.exp(2j * torch.pi * k1 / N[0])
#     E2 = 1 - torch.exp(2j * torch.pi * k2 / N[1])
#     E3 = 1 - torch.exp(2j * torch.pi * k3 / N[2])

#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)

#     s2 = torch.zeros(N)
    
#     # Allocate memory for first order gradient
#     s1_1 = torch.zeros(N)
#     s1_2 = torch.zeros(N)
#     s1_3 = torch.zeros(N)
#     z1_1 = torch.zeros(N)
#     z1_2 = torch.zeros(N)
#     z1_3 = torch.zeros(N)

#     # Allocate memory for symmetrized gradient
#     s0_1 = torch.zeros(N)
#     s0_2 = torch.zeros(N)
#     s0_3 = torch.zeros(N)
#     s0_4 = torch.zeros(N)
#     s0_5 = torch.zeros(N)
#     s0_6 = torch.zeros(N)
#     z0_1 = torch.zeros(N)
#     z0_2 = torch.zeros(N)
#     z0_3 = torch.zeros(N)
#     z0_4 = torch.zeros(N)
#     z0_5 = torch.zeros(N)
#     z0_6 = torch.zeros(N)

#     x = torch.zeros(N)
#     zero = torch.tensor(0)
    
#     # Move variables to GPU    
#     torch.cuda.empty_cache()
    
#     if isNonlinear:
#         phase = phase.to(device)
#     else:
#         FWy = FWy.to(device)
    
#     z1_1 = z1_1.to(device)
#     z1_2 = z1_2.to(device)
#     z1_3 = z1_3.to(device)
    
#     z0_1 = z0_1.to(device)
#     z0_2 = z0_2.to(device)
#     z0_3 = z0_3.to(device)
#     z0_4 = z0_4.to(device)
#     z0_5 = z0_5.to(device)
#     z0_6 = z0_6.to(device)
    
#     s1_1 = s1_1.to(device)
#     s1_2 = s1_2.to(device)
#     s1_3 = s1_3.to(device)
    
#     s0_1 = s0_1.to(device)
#     s0_2 = s0_2.to(device)
#     s0_3 = s0_3.to(device)
#     s0_4 = s0_4.to(device)
#     s0_5 = s0_5.to(device)
#     s0_6 = s0_6.to(device)      
    
#     x = x.to(device)
#     Kernel = Kernel.to(device)
    
#     z2 = z2.to(device)
#     s2 = s2.to(device)
    
#     tolUpdate = tolUpdate.to(device)
    
#     E1 = E1.to(device)
#     E2 = E2.to(device)
#     E3 = E3.to(device)
    
#     regWeight = regWeight.to(device)
#     mu1 = mu1.to(device)
#     mu2 = mu2.to(device)
#     mu0 = mu0.to(device)
#     weight = weight.to(device) # <-- Moved weight to device
#     W = W.to(device)
#     alpha1 = alpha1.to(device)
#     alpha0 = alpha0.to(device)
#     zero = zero.to(device)

#     # --- NRMSE History Initialization ---
#     chi_cosmos_gt = params.get("chi_cosmos", None)
#     rmse_history = []
#     chi_cosmos_tensor = None
#     if chi_cosmos_gt is not None:
#         chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
#         # mask_for_rmse is already defined and on device
#         masked_true = mask_for_rmse * chi_cosmos_tensor
#         norm_true = torch.norm(masked_true) + 1e-8
#     # --- End NRMSE Initialization ---
    
#     Et1, Et2, Et3 = torch.conj(E1), torch.conj(E2), torch.conj(E3)
#     Kt = torch.conj(Kernel)
    
#     E1tE1 = Et1 * E1
#     E2tE2 = Et2 * E2
#     E3tE3 = Et3 * E3
#     mu0_over_2_E1tE2 = mu0 / 2 * Et1 * E2
#     mu0_over_2_E1tE3 = mu0 / 2 * Et1 * E3
#     mu0_over_2_E2tE3 = mu0 / 2 * Et2 * E3
    
#     KtK_mu1_E_sos = mu2 * Kt * Kernel + mu1 * (E1tE1 + E2tE2 + E3tE3)
#     mu1I_mu0_E_wsos1 = mu1 + mu0 * (E1tE1 + (E2tE2 + E3tE3) / 2)
#     mu1I_mu0_E_wsos2 = mu1 + mu0 * (E1tE1 / 2 + E2tE2 + E3tE3 / 2)
#     mu1I_mu0_E_wsos3 = mu1 + mu0 * ((E1tE1 + E2tE2) / 2 + E3tE3)

#     del E1tE1  
#     del E2tE2
#     del E3tE3
    
#     # Precomputation for Cramer's Rule
#     a1 = KtK_mu1_E_sos
#     a2 = mu1I_mu0_E_wsos1
#     a3 = mu1I_mu0_E_wsos2
#     a4 = mu1I_mu0_E_wsos3
#     a5 = -mu1 * E1
#     a6 = -mu1 * E2
#     a7 = mu0_over_2_E1tE2
#     a8 = -mu1 * E3
#     a9 = mu0_over_2_E1tE3
#     a10 = mu0_over_2_E2tE3
    
#     a5t = torch.conj(a5)
#     a6t = torch.conj(a6)
#     a7t = torch.conj(a7)
#     a8t = torch.conj(a8)
#     a9t = torch.conj(a9)
#     a10t = torch.conj(a10)
    
#     # Free GPU memory for specific variables
#     del KtK_mu1_E_sos
#     del mu1I_mu0_E_wsos1
#     del mu1I_mu0_E_wsos2
#     del mu1I_mu0_E_wsos3
    
#     # For x
#     D11 = a2 * a3 * a4 + a7t * a9 * a10t + a7 * a9t * a10 - a3 * a9 * a9t - a2 * a10 * a10t - a4 * a7 * a7t
#     D21 = a3 * a4 * a5t + a6t * a9 * a10t + a7 * a8t * a10 - a3 * a8t * a9 - a5t * a10 * a10t - a4 * a6t * a7
#     D31 = a4 * a5t * a7t + a6t * a9 * a9t + a2 * a8t * a10 - a7t * a8t * a9 - a5t * a9t * a10 - a2 * a4 * a6t
#     D41 = a5t * a7t * a10t + a6t * a7 * a9t + a2 * a3 * a8t - a7 * a7t * a8t - a3 * a5t * a9t - a2 * a6t * a10t

#     # For vx
#     D12 = a3 * a4 * a5 + a7t * a8 * a10t + a6 * a9t * a10 - a3 * a8 * a9t - a5 * a10 * a10t - a4 * a6 * a7t
#     D22 = a1 * a3 * a4 + a6t * a8 * a10t + a6 * a8t * a10 - a3 * a8 * a8t - a1 * a10 * a10t - a4 * a6 * a6t
#     D32 = a1 * a4 * a7t + a6t * a8 * a9t + a5 * a8t * a10 - a7t * a8 * a8t - a1 * a9t * a10 - a4 * a5 * a6t
#     D42 = a1 * a7t * a10t + a6 * a6t * a9t + a3 * a5 * a8t - a6 * a7t * a8t - a1 * a3 * a9t - a5 * a6t * a10t

#     # For vy
#     D13 = a4 * a5 * a7 + a2 * a8 * a10t + a6 * a9 * a9t - a7 * a8 * a9t - a5 * a9 * a10t - a2 * a4 * a6
#     D23 = a1 * a4 * a7 + a5t * a8 * a10t + a6 * a8t * a9 - a7 * a8 * a8t - a1 * a9 * a10t - a4 * a5t * a6
#     D33 = a1 * a2 * a4 + a5t * a8 * a9t + a5 * a8t * a9 - a2 * a8 * a8t - a1 * a9 * a9t - a4 * a5 * a5t
#     D43 = a1 * a2 * a10t + a5t * a6 * a9t + a5 * a7 * a8t - a2 * a6 * a8t - a1 * a7 * a9t - a5 * a5t * a10t

#     # For vz
#     D14 = a5 * a7 * a10 + a2 * a3 * a8 + a6 * a7t * a9 - a7 * a7t * a8 - a3 * a5 * a9 - a2 * a6 * a10
#     D24 = a1 * a7 * a10 + a3 * a5t * a8 + a6 * a6t * a9 - a6t * a7 * a8 - a1 * a3 * a9 - a5t * a6 * a10
#     D34 = a1 * a2 * a10 + a5t * a7t * a8 + a5 * a6t * a9 - a2 * a6t * a8 - a1 * a7t * a9 - a5 * a5t * a10
#     D44 = a1 * a2 * a3 + a5t * a6 * a7t + a5 * a6t * a7 - a2 * a6 * a6t - a1 * a7 * a7t - a3 * a5 * a5t

#     det_A = a1 * D11 - a5 * D21 + a6 * D31 - a8 * D41
#     det_Ainv = 1 / (torch.finfo(torch.float64).eps + det_A)
    
#     # Free GPU memory for specific variables
#     del mu0_over_2_E1tE2
#     del mu0_over_2_E1tE3
#     del mu0_over_2_E2tE3
#     del a1
#     del a2
#     del a3
#     del a4
#     del a5
#     del a6
#     del a7
#     del a8
#     del a9
#     del a10
#     del a6t
#     del a7t
#     del a8t
#     del a9t
#     del a10t
#     del det_A
    
#     reg_time = time.time()
    
#     for t in range(maxOuterIter):
            
#         # Update x and v: susceptibility estimate
#         F_z0_minus_s0_1 = torch.fft.fftn(z0_1 - s0_1)
#         F_z0_minus_s0_2 = torch.fft.fftn(z0_2 - s0_2)
#         F_z0_minus_s0_3 = torch.fft.fftn(z0_3 - s0_3)
#         F_z0_minus_s0_4 = torch.fft.fftn(z0_4 - s0_4)
#         F_z0_minus_s0_5 = torch.fft.fftn(z0_5 - s0_5)
#         F_z0_minus_s0_6 = torch.fft.fftn(z0_6 - s0_6)
    
#         F_z1_minus_s1_1 = torch.fft.fftn(z1_1 - s1_1)
#         F_z1_minus_s1_2 = torch.fft.fftn(z1_2 - s1_2)
#         F_z1_minus_s1_3 = torch.fft.fftn(z1_3 - s1_3)
        
#         mu2_Kt = mu2 * Kt
#         Et1_F_z1_minus_s1_1 = Et1 * F_z1_minus_s1_1
#         Et2_F_z1_minus_s1_2 = Et2 * F_z1_minus_s1_2
#         Et3_F_z1_minus_s1_3 = Et3 * F_z1_minus_s1_3
        
#         rhs1 = mu2_Kt * (z2 - s2) + mu1 * (Et1_F_z1_minus_s1_1 + Et2_F_z1_minus_s1_2 + Et3_F_z1_minus_s1_3)
#         rhs2 = -mu1 * F_z1_minus_s1_1 + mu0 * (Et1 * F_z0_minus_s0_1 + Et2 * F_z0_minus_s0_4 + Et3 * F_z0_minus_s0_5)
#         rhs3 = -mu1 * F_z1_minus_s1_2 + mu0 * (Et2 * F_z0_minus_s0_2 + Et1 * F_z0_minus_s0_4 + Et3 * F_z0_minus_s0_6)
#         rhs4 = -mu1 * F_z1_minus_s1_3 + mu0 * (Et3 * F_z0_minus_s0_3 + Et1 * F_z0_minus_s0_5 + Et2 * F_z0_minus_s0_6)
            
#         # Free GPU memory for specific variables
#         del F_z0_minus_s0_1
#         del F_z0_minus_s0_2
#         del F_z0_minus_s0_3
#         del F_z0_minus_s0_4
#         del F_z0_minus_s0_5
#         del F_z0_minus_s0_6
#         del F_z1_minus_s1_1
#         del F_z1_minus_s1_2
#         del F_z1_minus_s1_3

#         # Cramer's rul
#         Fx = (rhs1 * D11 - rhs2 * D21 + rhs3 * D31 - rhs4 * D41) * det_Ainv
#         Fv1 = (-rhs1 * D12 + rhs2 * D22 - rhs3 * D32 + rhs4 * D42) * det_Ainv
#         Fv2 = (rhs1 * D13 - rhs2 * D23 + rhs3 * D33 - rhs4 * D43) * det_Ainv
#         Fv3 = (-rhs1 * D14 + rhs2 * D24 - rhs3 * D34 + rhs4 * D44) * det_Ainv
            
#         # Free GPU memory for specific variables
#         del rhs1
#         del rhs2
#         del rhs3
#         del rhs4
            
#         x_prev = x
#         x = torch.real(torch.fft.ifftn(Fx))
#         v1 = torch.real(torch.fft.ifftn(Fv1))
#         v2 = torch.real(torch.fft.ifftn(Fv2))
#         v3 = torch.real(torch.fft.ifftn(Fv3))

#         # --- NRMSE Calculation ---
#         if chi_cosmos_tensor is not None:
#             # Calculate masked error norm
#             masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
#             norm_error = torch.norm(masked_error)
            
#             # Calculate NRMSE percentage
#             nrmse = 100 * norm_error / norm_true
#             rmse_history.append(nrmse.item())
#         # --- End NRMSE Calculation ---
    
#         x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        
#         if chi_cosmos_tensor is not None:
#             print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
#         else:
#             print(f'Iter: {t+1}   Update: {x_update:.6f}')
            
#         if x_update < tolUpdate:
#             print("\nEarly stopping reached.")
#             break
        
#         if t < maxOuterIter:
        
#             # Update z: gradient variable
#             Fx = torch.fft.fftn(x)  # Extra step to increase stability from the proximal step
#             Fv1 = torch.fft.fftn(v1)
#             Fv2 = torch.fft.fftn(v2)
#             Fv3 = torch.fft.fftn(v3)
                
#             # Compute gradients for z0 and z1 update
#             Dx1 = torch.real(torch.fft.ifftn(E1 * Fx))
#             Dx2 = torch.real(torch.fft.ifftn(E2 * Fx))
#             Dx3 = torch.real(torch.fft.ifftn(E3 * Fx))
        
#             E_v1 = torch.real(torch.fft.ifftn(E1 * Fv1))
#             E_v2 = torch.real(torch.fft.ifftn(E2 * Fv2))
#             E_v3 = torch.real(torch.fft.ifftn(E3 * Fv3))
#             E_v4 = torch.real(torch.fft.ifftn(E1 * Fv2 + E2 * Fv1)) / 2
#             E_v5 = torch.real(torch.fft.ifftn(E1 * Fv3 + E3 * Fv1)) / 2
#             E_v6 = torch.real(torch.fft.ifftn(E2 * Fv3 + E3 * Fv2)) / 2
            
#             # Free GPU memory for specific variables
#             del Fv1
#             del Fv2
#             del Fv3
            
#             # Update z0: Symm grad
#             z0_1 = torch.max(torch.abs(E_v1 + s0_1) - alpha0_over_mu0, zero) * torch.sign(E_v1 + s0_1)
#             z0_2 = torch.max(torch.abs(E_v2 + s0_2) - alpha0_over_mu0, zero) * torch.sign(E_v2 + s0_2)
#             z0_3 = torch.max(torch.abs(E_v3 + s0_3) - alpha0_over_mu0, zero) * torch.sign(E_v3 + s0_3)
#             z0_4 = torch.max(torch.abs(E_v4 + s0_4) - alpha0_over_mu0, zero) * torch.sign(E_v4 + s0_4)
#             z0_5 = torch.max(torch.abs(E_v5 + s0_5) - alpha0_over_mu0, zero) * torch.sign(E_v5 + s0_5)
#             z0_6 = torch.max(torch.abs(E_v6 + s0_6) - alpha0_over_mu0, zero) * torch.sign(E_v6 + s0_6)
                
#             # Update z1: Grad
#             z1_1 = torch.max(torch.abs(Dx1 - v1 + s1_1) - regWeight[..., 0] * alpha_over_mu, zero) * torch.sign(Dx1 - v1 + s1_1)
#             z1_2 = torch.max(torch.abs(Dx2 - v2 + s1_2) - regWeight[..., 1] * alpha_over_mu, zero) * torch.sign(Dx2 - v2 + s1_2)
#             z1_3 = torch.max(torch.abs(Dx3 - v3 + s1_3) - regWeight[..., 2] * alpha_over_mu, zero) * torch.sign(Dx3 - v3 + s1_3)
            
#             # Update z2 and s2: data consistency
#             rhs_z2 = mu2 * torch.real(fft.ifftn(Kernel * Fx + s2))
#             if isNonlinear:
#                 # Update z2 and s2: data consistency
#                 z2 = rhs_z2 / mu2
                    
#                 # Newton-Raphson method
#                 delta = float('inf')
#                 inn = 0
#                 while (delta > tolDelta and inn < 50):
#                     inn = inn + 1
#                     norm_old = torch.norm(z2.reshape(-1))
                
#                     update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
                
#                     z2 -= update
#                     delta = torch.norm(update.reshape(-1)) / norm_old
                    
#                 z2 = fft.fftn(z2)
                
#                 # Free GPU memory for specific variables
#                 del rhs_z2
#                 del update
#             else:
#                 z2 = FWy + torch.fft.fftn(rhs_z2 / (W + mu2))
            
#             s2 = s2 + Kernel * Fx - z2
                    
#             # Update s0 and s1
#             s0_1 = s0_1 + E_v1 - z0_1
#             s0_2 = s0_2 + E_v2 - z0_2
#             s0_3 = s0_3 + E_v3 - z0_3
#             s0_4 = s0_4 + E_v4 - z0_4
#             s0_5 = s0_5 + E_v5 - z0_5
#             s0_6 = s0_6 + E_v6 - z0_6
        
#             s1_1 = s1_1 + Dx1 - v1 - z1_1
#             s1_2 = s1_2 + Dx2 - v2 - z1_2
#             s1_3 = s1_3 + Dx3 - v3 - z1_3
            
#             # Free GPU memory for specific variables
#             del Dx1
#             del Dx2
#             del Dx3
#             del E_v1
#             del E_v2
#             del E_v3
#             del E_v4
#             del E_v5
#             del E_v6
            
#     # Extract output values
#     end_time = time.time()
#     reg_time = end_time - reg_time
#     total_time = end_time - start_time
#     print(f'Regularization Time: {reg_time:.4f} seconds')
#     print(f'Total Execution Time: {total_time:.4f} seconds')
    
#     out = {
#         "x": x.cpu().detach().numpy().astype(np.float32), 
#         "iter": t+1, 
#         "time": total_time
#     }
    
#     if chi_cosmos_tensor is not None:
#         out["rmse_history"] = rmse_history
        
#     return out

import time
import torch
import numpy as np

def fansiTGV(params):
    """
    Total Generalized Variation (TGV) QSM regularization.
    
    Parameters:
    params['input']: local field map (phase)
    params['K']: dipole kernel
    params['alpha']: gradient penalty weight
    params['alpha0']: symmetrized gradient penalty weight (defaults to 2 * alpha)
    params['isNonlinear']: (bool) Switch between Linear and Nonlinear data consistency
    params['chi_cosmos']: (optional) Ground truth for NRMSE
    params['isLcurve']: (optional) boolean to compute residual and TGV norms for L-curve analysis.
    params['isXSIM']: (optional) boolean to compute XSIM at each iteration (default: False).
    params['phs_scale']: (optional) scale factor to bring x / chi_cosmos into the expected range for XSIM.
    
    Output:
    out['x']: susceptibility map
    out['iter']: iterations
    out['time']: total execution time
    out['update_history']: list of relative update (%) values per iteration.
    out['rmse_history']: NRMSE evolution
    out['xsim_history']: (if isXSIM) list of XSIM values per iteration.
    out['res_norm']: (if isLcurve) residual norm of the final solution.
    out['tgv_norm']: (if isLcurve) TGV norm of the final solution.
    """
    
    start_time = time.time()
    # Define device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Required parameters
    Kernel = torch.tensor(params["K"], dtype=torch.float32)
    phase = torch.tensor(params["input"], dtype=torch.float64)
    
    alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float32)
    alpha0 = params.get("alpha0", 2 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)      
    N = phase.shape
    maxOuterIter = params.get("maxOuterIter", 150)

    # Load fidelity weight (e.g., magnitude)
    weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32)
    W = weight * weight
    
    # Load the separate mask for NRMSE calculation
    mask_np = params.get("mask", None)
    if mask_np is None:
        print("Warning: params['mask'] not provided for NRMSE. Using params['weight'] as mask.")
        mask_for_rmse = weight.to(device) # Fallback to using fidelity weight
    else:
        mask_for_rmse = torch.tensor(mask_np, dtype=torch.float32, device=device)

    mu1 = params.get("mu1", 100 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
    mu0 = params.get("mu0", 200 * alpha1).clone().detach().requires_grad_(False).to(torch.float32)
    mu2 = torch.tensor(params.get("mu2", 1), dtype=torch.float32)

    tolUpdate = torch.tensor(params.get("tolUpdate", 0.1))
    voxelSize = np.array(params.get("voxelSize", [[1,1,1]]))
    regWeight = torch.from_numpy(np.ones(N + (3,)))
    
    isNonlinear = params.get("isNonlinear", False)
    
    # --- L-curve & XSIM toggles added here ---
    is_lcurve = params.get("isLcurve", False)
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("phs_scale", 1.0), dtype=torch.float32).to(device)
    
    if isNonlinear:
        tolDelta = torch.tensor(params.get("tolDelta", 1E-6))
        z2 =  torch.fft.fftn(W * phase / (W + mu2))
    else:
        # Redefine variable for computational efficiency
        FWy = torch.fft.fftn(W * phase / (W + mu2))
        z2 = FWy  # start with something similar to the input phase, weighted to reduce noise
            
    alpha_over_mu = (alpha1 / mu1).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency
    alpha0_over_mu0 = (alpha0 / mu0).clone().detach().requires_grad_(False).to(torch.float32) # for efficiency
    
    # Precompute gradient-related matrices
    k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')
    #E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])) / voxelSize[0, 0]
    #E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])) / voxelSize[0, 1]
    #E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])) / voxelSize[0, 2]
    E1 = 1 - torch.exp(2j * torch.pi * k1 / N[0])
    E2 = 1 - torch.exp(2j * torch.pi * k2 / N[1])
    E3 = 1 - torch.exp(2j * torch.pi * k3 / N[2])

    E1 = E1.to(device)
    E2 = E2.to(device)
    E3 = E3.to(device)

    s2 = torch.zeros(N)
    
    # Allocate memory for first order gradient
    s1_1 = torch.zeros(N)
    s1_2 = torch.zeros(N)
    s1_3 = torch.zeros(N)
    z1_1 = torch.zeros(N)
    z1_2 = torch.zeros(N)
    z1_3 = torch.zeros(N)

    # Allocate memory for symmetrized gradient
    s0_1 = torch.zeros(N)
    s0_2 = torch.zeros(N)
    s0_3 = torch.zeros(N)
    s0_4 = torch.zeros(N)
    s0_5 = torch.zeros(N)
    s0_6 = torch.zeros(N)
    z0_1 = torch.zeros(N)
    z0_2 = torch.zeros(N)
    z0_3 = torch.zeros(N)
    z0_4 = torch.zeros(N)
    z0_5 = torch.zeros(N)
    z0_6 = torch.zeros(N)

    x = torch.zeros(N)
    zero = torch.tensor(0)
    
    # Move variables to GPU
    torch.cuda.empty_cache()

    # NOTE: phase must always be moved to device (not just for isNonlinear), since
    # the linear data-fidelity branch and the L-curve residual calculation both
    # reference it directly; previously it stayed on CPU when isNonlinear=False,
    # causing a device-mismatch error against the GPU tensors it is combined with.
    phase = phase.to(device)
    if not isNonlinear:
        FWy = FWy.to(device)

    z1_1 = z1_1.to(device)
    z1_2 = z1_2.to(device)
    z1_3 = z1_3.to(device)
    
    z0_1 = z0_1.to(device)
    z0_2 = z0_2.to(device)
    z0_3 = z0_3.to(device)
    z0_4 = z0_4.to(device)
    z0_5 = z0_5.to(device)
    z0_6 = z0_6.to(device)
    
    s1_1 = s1_1.to(device)
    s1_2 = s1_2.to(device)
    s1_3 = s1_3.to(device)
    
    s0_1 = s0_1.to(device)
    s0_2 = s0_2.to(device)
    s0_3 = s0_3.to(device)
    s0_4 = s0_4.to(device)
    s0_5 = s0_5.to(device)
    s0_6 = s0_6.to(device)      
    
    x = x.to(device)
    Kernel = Kernel.to(device)
    
    z2 = z2.to(device)
    s2 = s2.to(device)
    
    tolUpdate = tolUpdate.to(device)
    
    E1 = E1.to(device)
    E2 = E2.to(device)
    E3 = E3.to(device)
    
    regWeight = regWeight.to(device)
    mu1 = mu1.to(device)
    mu2 = mu2.to(device)
    mu0 = mu0.to(device)
    weight = weight.to(device) # <-- Moved weight to device
    W = W.to(device)
    alpha1 = alpha1.to(device)
    alpha0 = alpha0.to(device)
    zero = zero.to(device)

    # --- NRMSE History Initialization ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32).to(device)
        # mask_for_rmse is already defined and on device
        masked_true = mask_for_rmse * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8
    # --- End NRMSE Initialization ---

    # --- XSIM History Initialization ---
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_buffer = torch.zeros(maxOuterIter, device=device, dtype=torch.float32)
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask_for_rmse.bool()
    # --- End XSIM Initialization ---
    
    Et1, Et2, Et3 = torch.conj(E1), torch.conj(E2), torch.conj(E3)
    Kt = torch.conj(Kernel)
    
    E1tE1 = Et1 * E1
    E2tE2 = Et2 * E2
    E3tE3 = Et3 * E3
    mu0_over_2_E1tE2 = mu0 / 2 * Et1 * E2
    mu0_over_2_E1tE3 = mu0 / 2 * Et1 * E3
    mu0_over_2_E2tE3 = mu0 / 2 * Et2 * E3
    
    KtK_mu1_E_sos = mu2 * Kt * Kernel + mu1 * (E1tE1 + E2tE2 + E3tE3)
    mu1I_mu0_E_wsos1 = mu1 + mu0 * (E1tE1 + (E2tE2 + E3tE3) / 2)
    mu1I_mu0_E_wsos2 = mu1 + mu0 * (E1tE1 / 2 + E2tE2 + E3tE3 / 2)
    mu1I_mu0_E_wsos3 = mu1 + mu0 * ((E1tE1 + E2tE2) / 2 + E3tE3)

    del E1tE1  
    del E2tE2
    del E3tE3
    
    # Precomputation for Cramer's Rule
    a1 = KtK_mu1_E_sos
    a2 = mu1I_mu0_E_wsos1
    a3 = mu1I_mu0_E_wsos2
    a4 = mu1I_mu0_E_wsos3
    a5 = -mu1 * E1
    a6 = -mu1 * E2
    a7 = mu0_over_2_E1tE2
    a8 = -mu1 * E3
    a9 = mu0_over_2_E1tE3
    a10 = mu0_over_2_E2tE3
    
    a5t = torch.conj(a5)
    a6t = torch.conj(a6)
    a7t = torch.conj(a7)
    a8t = torch.conj(a8)
    a9t = torch.conj(a9)
    a10t = torch.conj(a10)
    
    # Free GPU memory for specific variables
    del KtK_mu1_E_sos
    del mu1I_mu0_E_wsos1
    del mu1I_mu0_E_wsos2
    del mu1I_mu0_E_wsos3
    
    # For x
    D11 = a2 * a3 * a4 + a7t * a9 * a10t + a7 * a9t * a10 - a3 * a9 * a9t - a2 * a10 * a10t - a4 * a7 * a7t
    D21 = a3 * a4 * a5t + a6t * a9 * a10t + a7 * a8t * a10 - a3 * a8t * a9 - a5t * a10 * a10t - a4 * a6t * a7
    D31 = a4 * a5t * a7t + a6t * a9 * a9t + a2 * a8t * a10 - a7t * a8t * a9 - a5t * a9t * a10 - a2 * a4 * a6t
    D41 = a5t * a7t * a10t + a6t * a7 * a9t + a2 * a3 * a8t - a7 * a7t * a8t - a3 * a5t * a9t - a2 * a6t * a10t

    # For vx
    D12 = a3 * a4 * a5 + a7t * a8 * a10t + a6 * a9t * a10 - a3 * a8 * a9t - a5 * a10 * a10t - a4 * a6 * a7t
    D22 = a1 * a3 * a4 + a6t * a8 * a10t + a6 * a8t * a10 - a3 * a8 * a8t - a1 * a10 * a10t - a4 * a6 * a6t
    D32 = a1 * a4 * a7t + a6t * a8 * a9t + a5 * a8t * a10 - a7t * a8 * a8t - a1 * a9t * a10 - a4 * a5 * a6t
    D42 = a1 * a7t * a10t + a6 * a6t * a9t + a3 * a5 * a8t - a6 * a7t * a8t - a1 * a3 * a9t - a5 * a6t * a10t

    # For vy
    D13 = a4 * a5 * a7 + a2 * a8 * a10t + a6 * a9 * a9t - a7 * a8 * a9t - a5 * a9 * a10t - a2 * a4 * a6
    D23 = a1 * a4 * a7 + a5t * a8 * a10t + a6 * a8t * a9 - a7 * a8 * a8t - a1 * a9 * a10t - a4 * a5t * a6
    D33 = a1 * a2 * a4 + a5t * a8 * a9t + a5 * a8t * a9 - a2 * a8 * a8t - a1 * a9 * a9t - a4 * a5 * a5t
    D43 = a1 * a2 * a10t + a5t * a6 * a9t + a5 * a7 * a8t - a2 * a6 * a8t - a1 * a7 * a9t - a5 * a5t * a10t

    # For vz
    D14 = a5 * a7 * a10 + a2 * a3 * a8 + a6 * a7t * a9 - a7 * a7t * a8 - a3 * a5 * a9 - a2 * a6 * a10
    D24 = a1 * a7 * a10 + a3 * a5t * a8 + a6 * a6t * a9 - a6t * a7 * a8 - a1 * a3 * a9 - a5t * a6 * a10
    D34 = a1 * a2 * a10 + a5t * a7t * a8 + a5 * a6t * a9 - a2 * a6t * a8 - a1 * a7t * a9 - a5 * a5t * a10
    D44 = a1 * a2 * a3 + a5t * a6 * a7t + a5 * a6t * a7 - a2 * a6 * a6t - a1 * a7 * a7t - a3 * a5 * a5t

    det_A = a1 * D11 - a5 * D21 + a6 * D31 - a8 * D41
    det_Ainv = 1 / (torch.finfo(torch.float64).eps + det_A)
    
    # Free GPU memory for specific variables
    del mu0_over_2_E1tE2
    del mu0_over_2_E1tE3
    del mu0_over_2_E2tE3
    del a1
    del a2
    del a3
    del a4
    del a5
    del a6
    del a7
    del a8
    del a9
    del a10
    del a6t
    del a7t
    del a8t
    del a9t
    del a10t
    del det_A
    
    reg_time = time.time()
    update_history = []

    for t in range(maxOuterIter):

        # Update x and v: susceptibility estimate
        F_z0_minus_s0_1 = torch.fft.fftn(z0_1 - s0_1)
        F_z0_minus_s0_2 = torch.fft.fftn(z0_2 - s0_2)
        F_z0_minus_s0_3 = torch.fft.fftn(z0_3 - s0_3)
        F_z0_minus_s0_4 = torch.fft.fftn(z0_4 - s0_4)
        F_z0_minus_s0_5 = torch.fft.fftn(z0_5 - s0_5)
        F_z0_minus_s0_6 = torch.fft.fftn(z0_6 - s0_6)
    
        F_z1_minus_s1_1 = torch.fft.fftn(z1_1 - s1_1)
        F_z1_minus_s1_2 = torch.fft.fftn(z1_2 - s1_2)
        F_z1_minus_s1_3 = torch.fft.fftn(z1_3 - s1_3)
        
        mu2_Kt = mu2 * Kt
        Et1_F_z1_minus_s1_1 = Et1 * F_z1_minus_s1_1
        Et2_F_z1_minus_s1_2 = Et2 * F_z1_minus_s1_2
        Et3_F_z1_minus_s1_3 = Et3 * F_z1_minus_s1_3
        
        rhs1 = mu2_Kt * (z2 - s2) + mu1 * (Et1_F_z1_minus_s1_1 + Et2_F_z1_minus_s1_2 + Et3_F_z1_minus_s1_3)
        rhs2 = -mu1 * F_z1_minus_s1_1 + mu0 * (Et1 * F_z0_minus_s0_1 + Et2 * F_z0_minus_s0_4 + Et3 * F_z0_minus_s0_5)
        rhs3 = -mu1 * F_z1_minus_s1_2 + mu0 * (Et2 * F_z0_minus_s0_2 + Et1 * F_z0_minus_s0_4 + Et3 * F_z0_minus_s0_6)
        rhs4 = -mu1 * F_z1_minus_s1_3 + mu0 * (Et3 * F_z0_minus_s0_3 + Et1 * F_z0_minus_s0_5 + Et2 * F_z0_minus_s0_6)
            
        # Free GPU memory for specific variables
        del F_z0_minus_s0_1
        del F_z0_minus_s0_2
        del F_z0_minus_s0_3
        del F_z0_minus_s0_4
        del F_z0_minus_s0_5
        del F_z0_minus_s0_6
        del F_z1_minus_s1_1
        del F_z1_minus_s1_2
        del F_z1_minus_s1_3

        # Cramer's rul
        Fx = (rhs1 * D11 - rhs2 * D21 + rhs3 * D31 - rhs4 * D41) * det_Ainv
        Fv1 = (-rhs1 * D12 + rhs2 * D22 - rhs3 * D32 + rhs4 * D42) * det_Ainv
        Fv2 = (rhs1 * D13 - rhs2 * D23 + rhs3 * D33 - rhs4 * D43) * det_Ainv
        Fv3 = (-rhs1 * D14 + rhs2 * D24 - rhs3 * D34 + rhs4 * D44) * det_Ainv
            
        # Free GPU memory for specific variables
        del rhs1
        del rhs2
        del rhs3
        del rhs4
            
        x_prev = x
        x = torch.real(torch.fft.ifftn(Fx))
        v1 = torch.real(torch.fft.ifftn(Fv1))
        v2 = torch.real(torch.fft.ifftn(Fv2))
        v3 = torch.real(torch.fft.ifftn(Fv3))

        # --- NRMSE & XSIM Calculation ---
        if chi_cosmos_tensor is not None:
            # Calculate masked error norm
            masked_error = mask_for_rmse * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            
            # Calculate NRMSE percentage
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # Calculate XSIM
            if isXSIM:
                mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_buffer[t] = mssim
        # --- End NRMSE & XSIM Calculation ---
    
        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())

        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_buffer[t]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')

        if x_update < tolUpdate:
            print("\nEarly stopping reached.")
            break
        
        if t < maxOuterIter:
        
            # Update z: gradient variable
            Fx = torch.fft.fftn(x)  # Extra step to increase stability from the proximal step
            Fv1 = torch.fft.fftn(v1)
            Fv2 = torch.fft.fftn(v2)
            Fv3 = torch.fft.fftn(v3)
                
            # Compute gradients for z0 and z1 update
            Dx1 = torch.real(torch.fft.ifftn(E1 * Fx))
            Dx2 = torch.real(torch.fft.ifftn(E2 * Fx))
            Dx3 = torch.real(torch.fft.ifftn(E3 * Fx))
        
            E_v1 = torch.real(torch.fft.ifftn(E1 * Fv1))
            E_v2 = torch.real(torch.fft.ifftn(E2 * Fv2))
            E_v3 = torch.real(torch.fft.ifftn(E3 * Fv3))
            E_v4 = torch.real(torch.fft.ifftn(E1 * Fv2 + E2 * Fv1)) / 2
            E_v5 = torch.real(torch.fft.ifftn(E1 * Fv3 + E3 * Fv1)) / 2
            E_v6 = torch.real(torch.fft.ifftn(E2 * Fv3 + E3 * Fv2)) / 2
            
            # Free GPU memory for specific variables
            del Fv1
            del Fv2
            del Fv3
            
            # Update z0: Symm grad
            z0_1 = torch.max(torch.abs(E_v1 + s0_1) - alpha0_over_mu0, zero) * torch.sign(E_v1 + s0_1)
            z0_2 = torch.max(torch.abs(E_v2 + s0_2) - alpha0_over_mu0, zero) * torch.sign(E_v2 + s0_2)
            z0_3 = torch.max(torch.abs(E_v3 + s0_3) - alpha0_over_mu0, zero) * torch.sign(E_v3 + s0_3)
            z0_4 = torch.max(torch.abs(E_v4 + s0_4) - alpha0_over_mu0, zero) * torch.sign(E_v4 + s0_4)
            z0_5 = torch.max(torch.abs(E_v5 + s0_5) - alpha0_over_mu0, zero) * torch.sign(E_v5 + s0_5)
            z0_6 = torch.max(torch.abs(E_v6 + s0_6) - alpha0_over_mu0, zero) * torch.sign(E_v6 + s0_6)
                
            # Update z1: Grad
            z1_1 = torch.max(torch.abs(Dx1 - v1 + s1_1) - regWeight[..., 0] * alpha_over_mu, zero) * torch.sign(Dx1 - v1 + s1_1)
            z1_2 = torch.max(torch.abs(Dx2 - v2 + s1_2) - regWeight[..., 1] * alpha_over_mu, zero) * torch.sign(Dx2 - v2 + s1_2)
            z1_3 = torch.max(torch.abs(Dx3 - v3 + s1_3) - regWeight[..., 2] * alpha_over_mu, zero) * torch.sign(Dx3 - v3 + s1_3)
            
            # Update z2 and s2: data consistency
            rhs_z2 = mu2 * torch.real(torch.fft.ifftn(Kernel * Fx + s2))
            
            if isNonlinear:
                # Update z2: data consistency with aligned Newton-Raphson method
                z2 = rhs_z2 / mu2
                delta = float('inf')
                inn = 0
                while (delta > tolDelta and inn < 10):
                    inn += 1
                    norm_old = torch.norm(z2.reshape(-1))
                    update = (W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2) / (W * torch.cos(z2 - phase) + mu2)
                    z2 = z2 - update
                    delta = torch.norm(update.reshape(-1)) / norm_old
                z2 = torch.fft.fftn(z2)
                
                del rhs_z2
                del update
            else:
                z2 = FWy + torch.fft.fftn(rhs_z2 / (W + mu2))
            
            s2 = s2 + Kernel * Fx - z2
                    
            # Update s0 and s1
            s0_1 = s0_1 + E_v1 - z0_1
            s0_2 = s0_2 + E_v2 - z0_2
            s0_3 = s0_3 + E_v3 - z0_3
            s0_4 = s0_4 + E_v4 - z0_4
            s0_5 = s0_5 + E_v5 - z0_5
            s0_6 = s0_6 + E_v6 - z0_6
        
            s1_1 = s1_1 + Dx1 - v1 - z1_1
            s1_2 = s1_2 + Dx2 - v2 - z1_2
            s1_3 = s1_3 + Dx3 - v3 - z1_3
            
            # Free GPU memory for specific variables
            del Dx1
            del Dx2
            del Dx3
            del E_v1
            del E_v2
            del E_v3
            del E_v4
            del E_v5
            del E_v6

    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        Fx = torch.fft.fftn(x)
        Kx = torch.real(torch.fft.ifftn(Kernel * Fx))
        
        if isNonlinear:
            res_norm = torch.sum(W * torch.abs(torch.exp(1j * Kx) - torch.exp(1j * phase))**2)
        else:
            res_norm = torch.sum(W * (Kx - phase)**2)

        # TGV Norm components (computed with the latest x, v1, v2, v3)
        Fv1 = torch.fft.fftn(v1)
        Fv2 = torch.fft.fftn(v2)
        Fv3 = torch.fft.fftn(v3)
        
        Dx1 = torch.real(torch.fft.ifftn(E1 * Fx))
        Dx2 = torch.real(torch.fft.ifftn(E2 * Fx))
        Dx3 = torch.real(torch.fft.ifftn(E3 * Fx))

        E_v1 = torch.real(torch.fft.ifftn(E1 * Fv1))
        E_v2 = torch.real(torch.fft.ifftn(E2 * Fv2))
        E_v3 = torch.real(torch.fft.ifftn(E3 * Fv3))
        E_v4 = torch.real(torch.fft.ifftn(E1 * Fv2 + E2 * Fv1)) / 2
        E_v5 = torch.real(torch.fft.ifftn(E1 * Fv3 + E3 * Fv1)) / 2
        E_v6 = torch.real(torch.fft.ifftn(E2 * Fv3 + E3 * Fv2)) / 2

        tgv_norm = (torch.sum(torch.abs(Dx1 - v1)) + torch.sum(torch.abs(Dx2 - v2)) + torch.sum(torch.abs(Dx3 - v3))) + \
                   (torch.sum(torch.abs(E_v1)) + torch.sum(torch.abs(E_v2)) + torch.sum(torch.abs(E_v3)) + \
                    2 * torch.sum(torch.abs(E_v4)) + 2 * torch.sum(torch.abs(E_v5)) + 2 * torch.sum(torch.abs(E_v6)))
    # --- End L-curve Calculations ---
            
    # Extract output values
    end_time = time.time()
    reg_time = end_time - reg_time
    total_time = end_time - start_time
    print(f'\nRegularization Time: {reg_time:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')
    
    out = {
        "x": x.cpu().detach().numpy().astype(np.float32),
        "iter": t+1,
        "time": total_time,
        "update_history": update_history
    }

    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_buffer[:t + 1].cpu().tolist()

    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["tgv_norm"] = tgv_norm.item()

    return out


# def WH_fansiTV(params):
#     """
#     Weak Harmonics (WH) QSM with Total Variation regularization.
#     Combines Linear (wTV) and Nonlinear (nlTV) formulations.
    
#     Parameters:
#     params['input']: local field map (phase)
#     params['K']: dipole kernel
#     params['alpha']: gradient penalty weight
#     params['beta']: harmonic constraint weight (default=150)
#     params['muh']: harmonic consistency weight (default=beta/50)
#     params['isNonlinear']: (bool) Switch between Linear and Nonlinear TV
#     params['chi_cosmos']: (optional) Ground truth for NRMSE
    
#     Output:
#     out['x']: susceptibility map
#     out['phi']: harmonic background field
#     out['iter']: iterations
#     out['rmse_history']: NRMSE evolution
#     """
    
#     start_time = time.time()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # --- Parse Inputs ---
#     phase = torch.tensor(params["input"], dtype=torch.float64).to(device) # Keep double for precision input
#     Kernel = torch.tensor(params["K"], dtype=torch.float32).to(device)
#     N = phase.shape
    
#     # Weights and Parameters
#     alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float32, device=device)
#     mu1 = torch.tensor(params.get("mu1", 100.0 * alpha1), dtype=torch.float32, device=device)
#     mu2 = torch.tensor(params.get("mu2", 1.0), dtype=torch.float32, device=device)
    
#     # Weak Harmonics Parameters
#     beta = torch.tensor(params.get("beta", 150.0), dtype=torch.float32, device=device)
#     muh = torch.tensor(params.get("muh", 5.0), dtype=torch.float32, device=device)
    
#     maxOuterIter = params.get("maxOuterIter", 150)
#     tolUpdate = torch.tensor(params.get("tolUpdate", 0.1), device=device)
    
#     # Mode selector
#     isNonlinear = params.get("isNonlinear", False)
#     tolDelta = torch.tensor(params.get("tolDelta", 1e-6), device=device)

#     # Weights Handling
#     # MATLAB: W = W.*W
#     raw_weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32, device=device)
#     W = raw_weight * raw_weight 
    
#     # Mask Handling
#     mask_np = params.get("mask", None)
#     if mask_np is None:
#         # MATLAB: mask = W > 0
#         mask = (W > 0).float()
#     else:
#         mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    
#     # Regularization Weights
#     regWeight = torch.from_numpy(np.ones(N + (3,))).float().to(device) # Simplified for standard case

#     # Pre-calculations
#     # Linear Case Pre-calculation for z2
#     Wy = (W * phase) / (W + mu2)
#     alpha_over_mu = alpha1 / mu1
    
#     # --- Variable Initialization ---
#     z_dx = torch.zeros(N, dtype=torch.float32, device=device)
#     z_dy = torch.zeros(N, dtype=torch.float32, device=device)
#     z_dz = torch.zeros(N, dtype=torch.float32, device=device)

#     s_dx = torch.zeros(N, dtype=torch.float32, device=device)
#     s_dy = torch.zeros(N, dtype=torch.float32, device=device)
#     s_dz = torch.zeros(N, dtype=torch.float32, device=device)

#     x = torch.zeros(N, dtype=torch.float32, device=device)
    
#     # Weak Harmonics Variables
#     #phi_h = torch.zeros(N, dtype=torch.float32, device=device)
#     if "phi_init" in params:
#         phi_h = torch.tensor(params["phi_init"], dtype=torch.float32, device=device)
#         print("WH_fansiTV: Using provided background field initialization.")
#     else:
#         phi_h = torch.zeros(N, dtype=torch.float32, device=device)
#     z_h = torch.zeros(N, dtype=torch.float32, device=device)
#     s_h = torch.zeros(N, dtype=torch.float32, device=device)

#     # z2 Initialization
#     # MATLAB logic: if isPrecond z2=Wy (Linear) or z2=phase.*W (Nonlinear)
#     # We default to a smart start based on mode
#     if isNonlinear:
#         z2 = phase.float() * W # Approximation of phase
#     else:
#         z2 = Wy.clone()
        
#     s2 = torch.zeros(N, dtype=torch.float32, device=device)

#     # --- Operators (FAITHFUL TRANSLATION FIX) ---
#     # Use indexing='ij' to match MATLAB ndgrid
#     k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')
    
#     # NOTE: Removed division by voxelSize to match MATLAB implementation exactly
#     E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])).to(device)
#     E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])).to(device)
#     E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])).to(device)

#     E1t = torch.conj(E1)
#     E2t = torch.conj(E2)
#     E3t = torch.conj(E3)

#     EE2 = E1t * E1 + E2t * E2 + E3t * E3
#     Kt = torch.conj(Kernel)
    
#     zero = torch.tensor(0.0, device=device)
#     eps = torch.tensor(torch.finfo(torch.float32).eps, device=device)

#     # --- NRMSE Setup ---
#     chi_cosmos_gt = params.get("chi_cosmos", None)
#     rmse_history = []
#     chi_cosmos_tensor = None
#     if chi_cosmos_gt is not None:
#         chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32, device=device)
#         # Use the provided mask or weight for NRMSE region
#         masked_true = mask * chi_cosmos_tensor
#         norm_true = torch.norm(masked_true) + 1e-8

#     # --- Main Optimization Loop ---
#     print(f"{'Iter':<5} {'SubIter':<10} {'Update':<10} {'Delta/NRMSE':<15}")
    
#     regTime = time.time()
    
#     for t in range(maxOuterIter):
        
#         # 1. Update x: susceptibility estimate
#         tx = E1t * fft.fftn(z_dx - s_dx)
#         ty = E2t * fft.fftn(z_dy - s_dy)
#         tz = E3t * fft.fftn(z_dz - s_dz)
        
#         # Note: phi_h is subtracted in the data term
#         Dt_kspace = Kt * fft.fftn(z2 - s2 - phi_h)
        
#         x_prev = x.clone()
        
#         numerator = mu1 * (tx + ty + tz) + Dt_kspace
#         denominator = eps + mu2 * torch.abs(Kernel)**2 + mu1 * EE2
        
#         x = mask * torch.real(fft.ifftn(numerator / denominator))
        
#         del tx, ty, tz, numerator, denominator, Dt_kspace
        
#         # --- Convergence & NRMSE ---
#         x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        
#         current_metric = 0.0
#         if chi_cosmos_tensor is not None:
#             masked_error = mask * (x - chi_cosmos_tensor)
#             nrmse = 100 * torch.norm(masked_error) / norm_true
#             rmse_history.append(nrmse.item())
#             current_metric = nrmse.item()
        
#         if x_update < tolUpdate or torch.isnan(x_update):
#             print(f"{t+1:<5} {'-':<10} {x_update.item():<10.4f} {'Stopping':<15}")
#             break

#         if t < maxOuterIter:
#             # 2. Update z: gradient variable (Soft Thresholding)
#             Fx = fft.fftn(x)
#             x_dx = torch.real(fft.ifftn(E1 * Fx))
#             x_dy = torch.real(fft.ifftn(E2 * Fx))
#             x_dz = torch.real(fft.ifftn(E3 * Fx))

#             z_dx = torch.maximum(torch.abs(x_dx + s_dx) - regWeight[:, :, :, 0] * alpha_over_mu, zero) * torch.sign(x_dx + s_dx)
#             z_dy = torch.maximum(torch.abs(x_dy + s_dy) - regWeight[:, :, :, 1] * alpha_over_mu, zero) * torch.sign(x_dy + s_dy)
#             z_dz = torch.maximum(torch.abs(x_dz + s_dz) - regWeight[:, :, :, 2] * alpha_over_mu, zero) * torch.sign(x_dz + s_dz)

#             # 3. Update s: Lagrange multiplier
#             s_dx = s_dx + x_dx - z_dx
#             s_dy = s_dy + x_dy - z_dy
#             s_dz = s_dz + x_dz - z_dz
            
#             del x_dx, x_dy, x_dz

#             # 4. Update z2: Data Consistency
#             # Common term: Kernel * Fx + s2 + phi_h
#             k_fx = fft.ifftn(Kernel * Fx)
#             term_common = torch.real(k_fx) + s2 + phi_h
#             rhs_z2 = mu2 * term_common
            
#             delta_val = 0.0
#             inn = 0
            
#             if isNonlinear:
#                 # Nonlinear Solver (Newton-Raphson)
#                 z2 = rhs_z2 / mu2
#                 delta = float('inf')
                
#                 while delta > tolDelta and inn < 50:
#                     inn += 1
#                     norm_old = torch.norm(z2)
                    
#                     # Calculate update
#                     num = W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2
#                     den = W * torch.cos(z2 - phase) + mu2
#                     update = num / den
                    
#                     z2 = z2 - update
#                     delta = torch.norm(update) / norm_old
                    
#                 delta_val = delta
#             else:
#                 # Linear Solver (Closed form)
#                 z2 = Wy + rhs_z2 / (W + mu2)

#             # 5. Update phi_h: Harmonic Background Field
#             # Solve: (muh*Lap^H*Lap + mu2)*phi = ...
#             # The MATLAB code solves this in K-space
            
#             # Terms for phi_h RHS
#             # 1. muh * Lap^H * (z_h - s_h) -> In kspace: muh * conj(EE2) * FFT(z_h - s_h)
#             term1 = muh * torch.conj(EE2) * fft.fftn(z_h - s_h)
            
#             # 2. mu2 * (z2 - s2) -> FFT
#             term2 = mu2 * fft.fftn(z2 - s2)
            
#             # 3. -mu2 * Kernel * Fx -> Already have Kernel*Fx, just need to negate
#             # Note: The MATLAB code re-calculates Fx logic implicitly or uses stored
#             term3 = -mu2 * Kernel * Fx
            
#             numerator_phi = term1 + term2 + term3
#             denominator_phi = eps + mu2 + muh * EE2 * torch.conj(EE2)
            
#             Fphi_h = numerator_phi / denominator_phi
#             phi_h = torch.real(fft.ifftn(Fphi_h))
            
#             # 6. Update z_h: Harmonic Variable
#             # z_h = muh * (Lap(phi) + s_h) / (muh + beta*mask)
#             lap_phi = torch.real(fft.ifftn(EE2 * Fphi_h))
#             z_h = muh * (lap_phi + s_h) / (muh + beta * mask)
            
#             # 7. Update Lagrange Multipliers s2 and s_h
#             s2 = s2 + torch.real(k_fx) - z2 + phi_h
#             s_h = s_h + lap_phi - z_h
            
#             del Fx, Fphi_h, lap_phi, k_fx, rhs_z2

#         # Print status
#         metric_str = f"{current_metric:.4f}%" if chi_cosmos_tensor is not None else f"d={delta_val:.2e}"
#         print(f"{t+1:<5} {inn:<10} {x_update.item():<10.4f} {metric_str:<15}")

#     # --- Cleanup & Output ---
#     out = {
#         "x": x.cpu().detach().numpy().astype(np.float32),
#         "phi": phi_h.cpu().detach().numpy().astype(np.float32),
#         "iter": t + 1,
#         "time": time.time() - start_time
#     }
#     if chi_cosmos_tensor is not None:
#         out["rmse_history"] = rmse_history
        
#     return out


def WH_fansiTV(params):
    """
    Weak Harmonics (WH) QSM with Total Variation regularization.
    Combines Linear (wTV) and Nonlinear (nlTV) formulations.
    
    Parameters:
    params['input']: local field map (phase)
    params['K']: dipole kernel
    params['alpha']: gradient penalty weight
    params['beta']: harmonic constraint weight (default=150)
    params['muh']: harmonic consistency weight (default=beta/50)
    params['isNonlinear']: (bool) Switch between Linear and Nonlinear TV
    params['chi_cosmos']: (optional) Ground truth for NRMSE
    params['isLcurve']: (optional) boolean to compute residual and TV norms for L-curve analysis.
    params['isXSIM']: (optional) boolean to compute XSIM at each iteration (default: False).
    params['phs_scale']: (optional) scale factor to bring x / chi_cosmos into the expected range for XSIM.
    
    Output:
    out['x']: susceptibility map
    out['phi']: harmonic background field
    out['iter']: iterations
    out['time']: total elapsed time (including pre-calculations).
    out['update_history']: list of relative update (%) values per iteration.
    out['rmse_history']: NRMSE evolution
    out['xsim_history']: (if isXSIM) list of XSIM values per iteration.
    out['res_norm']: (if isLcurve) residual norm of the final solution.
    out['tv_norm']: (if isLcurve) Total Variation norm of the final solution.
    """
    
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Parse Inputs ---
    phase = torch.tensor(params["input"], dtype=torch.float64).to(device) # Keep double for precision input
    Kernel = torch.tensor(params["K"], dtype=torch.float32).to(device)
    N = phase.shape
    
    # Weights and Parameters
    alpha1 = torch.tensor(params.get("alpha", 1E-5), dtype=torch.float32, device=device)
    mu1 = torch.tensor(params.get("mu1", 100.0 * alpha1), dtype=torch.float32, device=device)
    mu2 = torch.tensor(params.get("mu2", 1.0), dtype=torch.float32, device=device)
    
    # Weak Harmonics Parameters
    beta = torch.tensor(params.get("beta", 150.0), dtype=torch.float32, device=device)
    muh = torch.tensor(params.get("muh", 5.0), dtype=torch.float32, device=device)
    
    maxOuterIter = params.get("maxOuterIter", 150)
    tolUpdate = torch.tensor(params.get("tolUpdate", 0.1), device=device)
    
    # Mode selector
    isNonlinear = params.get("isNonlinear", False)
    tolDelta = torch.tensor(params.get("tolDelta", 1e-6), device=device)

    # --- L-curve toggle added here ---
    is_lcurve = params.get("isLcurve", False)

    # --- XSIM toggle & scale added here ---
    isXSIM = params.get("isXSIM", False)
    scale = torch.tensor(params.get("phs_scale", 1.0), dtype=torch.float32).to(device)

    # Weights Handling
    # MATLAB: W = W.*W
    raw_weight = torch.tensor(params.get("weight", np.ones(N)), dtype=torch.float32, device=device)
    W = raw_weight * raw_weight 
    
    # Mask Handling
    mask_np = params.get("mask", None)
    if mask_np is None:
        # MATLAB: mask = W > 0
        mask = (W > 0).float()
    else:
        mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    
    # Regularization Weights
    regWeight = torch.from_numpy(np.ones(N + (3,))).float().to(device) # Simplified for standard case

    # Pre-calculations
    # Linear Case Pre-calculation for z2
    Wy = (W * phase) / (W + mu2)
    alpha_over_mu = alpha1 / mu1
    
    # --- Variable Initialization ---
    z_dx = torch.zeros(N, dtype=torch.float32, device=device)
    z_dy = torch.zeros(N, dtype=torch.float32, device=device)
    z_dz = torch.zeros(N, dtype=torch.float32, device=device)

    s_dx = torch.zeros(N, dtype=torch.float32, device=device)
    s_dy = torch.zeros(N, dtype=torch.float32, device=device)
    s_dz = torch.zeros(N, dtype=torch.float32, device=device)

    x = torch.zeros(N, dtype=torch.float32, device=device)
    
    # Weak Harmonics Variables
    #phi_h = torch.zeros(N, dtype=torch.float32, device=device)
    if "phi_init" in params:
        phi_h = torch.tensor(params["phi_init"], dtype=torch.float32, device=device)
        print("WH_fansiTV: Using provided background field initialization.")
    else:
        phi_h = torch.zeros(N, dtype=torch.float32, device=device)
    z_h = torch.zeros(N, dtype=torch.float32, device=device)
    s_h = torch.zeros(N, dtype=torch.float32, device=device)

    # z2 Initialization
    # MATLAB logic: if isPrecond z2=Wy (Linear) or z2=phase.*W (Nonlinear)
    # We default to a smart start based on mode
    if isNonlinear:
        z2 = phase.float() * W # Approximation of phase
    else:
        z2 = Wy.clone()
        
    s2 = torch.zeros(N, dtype=torch.float32, device=device)

    # --- Operators (FAITHFUL TRANSLATION FIX) ---
    # Use indexing='ij' to match MATLAB ndgrid
    k1, k2, k3 = torch.meshgrid(torch.arange(N[0]), torch.arange(N[1]), torch.arange(N[2]), indexing='ij')
    
    # NOTE: Removed division by voxelSize to match MATLAB implementation exactly
    E1 = (1 - torch.exp(2j * torch.pi * k1 / N[0])).to(device)
    E2 = (1 - torch.exp(2j * torch.pi * k2 / N[1])).to(device)
    E3 = (1 - torch.exp(2j * torch.pi * k3 / N[2])).to(device)

    E1t = torch.conj(E1)
    E2t = torch.conj(E2)
    E3t = torch.conj(E3)

    EE2 = E1t * E1 + E2t * E2 + E3t * E3
    Kt = torch.conj(Kernel)
    
    zero = torch.tensor(0.0, device=device)
    eps = torch.tensor(torch.finfo(torch.float32).eps, device=device)

    # --- NRMSE Setup ---
    chi_cosmos_gt = params.get("chi_cosmos", None)
    rmse_history = []
    chi_cosmos_tensor = None
    if chi_cosmos_gt is not None:
        chi_cosmos_tensor = torch.tensor(chi_cosmos_gt, dtype=torch.float32, device=device)
        # Use the provided mask or weight for NRMSE region
        masked_true = mask * chi_cosmos_tensor
        norm_true = torch.norm(masked_true) + 1e-8

    # --- XSIM History Initialization ---
    if isXSIM and chi_cosmos_tensor is not None:
        xsim_buffer = torch.zeros(maxOuterIter, device=device, dtype=torch.float32)
        xsim_kernels = _build_xsim_kernels(sw=(3, 3, 3), sigma=1.5, device=device, dtype=torch.float32)
        mask_bool_dev = mask.bool()
    # --- End XSIM Initialization ---

    # --- Main Optimization Loop ---
    regTime = time.time()
    update_history = []

    for t in range(maxOuterIter):

        # 1. Update x: susceptibility estimate
        tx = E1t * fft.fftn(z_dx - s_dx)
        ty = E2t * fft.fftn(z_dy - s_dy)
        tz = E3t * fft.fftn(z_dz - s_dz)
        
        # Note: phi_h is subtracted in the data term
        Dt_kspace = Kt * fft.fftn(z2 - s2 - phi_h)
        
        x_prev = x.clone()
        
        numerator = mu1 * (tx + ty + tz) + Dt_kspace
        denominator = eps + mu2 * torch.abs(Kernel)**2 + mu1 * EE2
        
        x = mask * torch.real(fft.ifftn(numerator / denominator))
        
        del tx, ty, tz, numerator, denominator, Dt_kspace
        
        # --- NRMSE Calculation ---
        if chi_cosmos_tensor is not None:
            masked_error = mask * (x - chi_cosmos_tensor)
            norm_error = torch.norm(masked_error)
            nrmse = 100 * norm_error / norm_true
            rmse_history.append(nrmse.item())

            # --- XSIM Calculation ---
            if isXSIM:
                mssim = compute_xsim_torch((x / scale) * mask_bool_dev, (chi_cosmos_tensor / scale) * mask_bool_dev, mask_bool_dev, xsim_kernels)
                xsim_buffer[t] = mssim
            # --- End XSIM Calculation ---
        # --- End NRMSE Calculation ---

        x_update = 100 * torch.norm(x - x_prev) / torch.norm(x)
        update_history.append(x_update.item())

        if chi_cosmos_tensor is not None:
            if isXSIM:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%   XSIM: {xsim_buffer[t]:.4f}')
            else:
                print(f'Iter: {t+1}   Update: {x_update:.6f}   NRMSE: {rmse_history[-1]:.4f}%')
        else:
            print(f'Iter: {t+1}   Update: {x_update:.6f}')

        if x_update < tolUpdate or torch.isnan(x_update):
            print("\nEarly stopping reached.")
            break

        if t < maxOuterIter:
            # 2. Update z: gradient variable (Soft Thresholding)
            Fx = fft.fftn(x)
            x_dx = torch.real(fft.ifftn(E1 * Fx))
            x_dy = torch.real(fft.ifftn(E2 * Fx))
            x_dz = torch.real(fft.ifftn(E3 * Fx))

            z_dx = torch.maximum(torch.abs(x_dx + s_dx) - regWeight[:, :, :, 0] * alpha_over_mu, zero) * torch.sign(x_dx + s_dx)
            z_dy = torch.maximum(torch.abs(x_dy + s_dy) - regWeight[:, :, :, 1] * alpha_over_mu, zero) * torch.sign(x_dy + s_dy)
            z_dz = torch.maximum(torch.abs(x_dz + s_dz) - regWeight[:, :, :, 2] * alpha_over_mu, zero) * torch.sign(x_dz + s_dz)

            # 3. Update s: Lagrange multiplier
            s_dx = s_dx + x_dx - z_dx
            s_dy = s_dy + x_dy - z_dy
            s_dz = s_dz + x_dz - z_dz
            
            del x_dx, x_dy, x_dz

            # 4. Update z2: Data Consistency
            # Common term: Kernel * Fx + s2 + phi_h
            k_fx = fft.ifftn(Kernel * Fx)
            term_common = torch.real(k_fx) + s2 + phi_h
            rhs_z2 = mu2 * term_common
            
            delta_val = 0.0
            inn = 0
            
            if isNonlinear:
                # Nonlinear Solver (Newton-Raphson)
                z2 = rhs_z2 / mu2
                delta = float('inf')
                
                while delta > tolDelta and inn < 10:
                    inn += 1
                    norm_old = torch.norm(z2)
                    
                    # Calculate update
                    num = W * torch.sin(z2 - phase) + mu2 * z2 - rhs_z2
                    den = W * torch.cos(z2 - phase) + mu2
                    update = num / den
                    
                    z2 = z2 - update
                    delta = torch.norm(update) / norm_old
                    
                delta_val = delta
            else:
                # Linear Solver (Closed form)
                z2 = Wy + rhs_z2 / (W + mu2)

            # 5. Update phi_h: Harmonic Background Field
            # Solve: (muh*Lap^H*Lap + mu2)*phi = ...
            # The MATLAB code solves this in K-space
            
            # Terms for phi_h RHS
            # 1. muh * Lap^H * (z_h - s_h) -> In kspace: muh * conj(EE2) * FFT(z_h - s_h)
            term1 = muh * torch.conj(EE2) * fft.fftn(z_h - s_h)
            
            # 2. mu2 * (z2 - s2) -> FFT
            term2 = mu2 * fft.fftn(z2 - s2)
            
            # 3. -mu2 * Kernel * Fx -> Already have Kernel*Fx, just need to negate
            # Note: The MATLAB code re-calculates Fx logic implicitly or uses stored
            term3 = -mu2 * Kernel * Fx
            
            numerator_phi = term1 + term2 + term3
            denominator_phi = eps + mu2 + muh * EE2 * torch.conj(EE2)
            
            Fphi_h = numerator_phi / denominator_phi
            phi_h = torch.real(fft.ifftn(Fphi_h))
            
            # 6. Update z_h: Harmonic Variable
            # z_h = muh * (Lap(phi) + s_h) / (muh + beta*mask)
            lap_phi = torch.real(fft.ifftn(EE2 * Fphi_h))
            z_h = muh * (lap_phi + s_h) / (muh + beta * mask)
            
            # 7. Update Lagrange Multipliers s2 and s_h
            s2 = s2 + torch.real(k_fx) - z2 + phi_h
            s_h = s_h + lap_phi - z_h
            
            del Fx, Fphi_h, lap_phi, k_fx, rhs_z2

    print('\n')

    # --- L-curve Norm Calculations Added Here ---
    if is_lcurve:
        Fx = fft.fftn(x)
        # Include harmonic background field (phi_h) in the data consistency calculation
        Kx = torch.real(fft.ifftn(Kernel * Fx)) + phi_h
        
        if isNonlinear:
            res_norm = torch.sum(W * torch.abs(torch.exp(1j * Kx) - torch.exp(1j * phase))**2)
        else:
            res_norm = torch.sum(W * (Kx - phase)**2)

        # TV norm in Fourier formulation using existing gradient operators
        x_dx = torch.real(fft.ifftn(E1 * Fx))
        x_dy = torch.real(fft.ifftn(E2 * Fx))
        x_dz = torch.real(fft.ifftn(E3 * Fx))
        tv_norm = torch.sum(torch.abs(x_dx)) + torch.sum(torch.abs(x_dy)) + torch.sum(torch.abs(x_dz))
    # --- End L-curve Calculations ---

    # --- Cleanup & Output ---
    end_time = time.time()
    reg_time = end_time - regTime
    total_time = end_time - start_time
    print(f'Regularization Time: {reg_time:.4f} seconds')
    print(f'Total Execution Time: {total_time:.4f} seconds')

    out = {
        "x": x.cpu().detach().numpy().astype(np.float32),
        "phi": phi_h.cpu().detach().numpy().astype(np.float32),
        "iter": t + 1,
        "time": total_time,
        "update_history": update_history
    }

    if chi_cosmos_tensor is not None:
        out["rmse_history"] = rmse_history
        if isXSIM:
            out["xsim_history"] = xsim_buffer[:t + 1].cpu().tolist()

    if is_lcurve:
        out["res_norm"] = res_norm.item()
        out["tv_norm"] = tv_norm.item()

    return out