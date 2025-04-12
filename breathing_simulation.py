# 🚀 Walter's Breathing Universe: Long-Time Military Test
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters (military-tested) ---
kappa = 1.0         
Lambda = 0.7        
g0 = 1.0            
gamma_pp = 5.0      

# --- Time Grid for Long Simulation ---
t_max = 500         # Long cosmic time
n_points = 5000     # Fine time resolution

# --- Breathing Field Dynamics ---
def d2phi_dt2(phi, chi):
    return (1/kappa) * (np.sin(phi) * (Lambda**4 - 0.5 * g0 * chi**2) - gamma_pp * Lambda**4 * chi * np.cos(phi))

def d2chi_dt2(phi, chi):
    return -g0 * (1 + np.cos(phi)) * chi + gamma_pp * Lambda**4 * np.sin(phi)

def breathing_odes(t, y):
    phi, dphi, chi, dchi = y
    ddphi = d2phi_dt2(phi, chi)
    ddchi = d2chi_dt2(phi, chi)
    return [dphi, ddphi, dchi, ddchi]

# --- Initial Conditions ---
phi0 = 0.7          # Born breathing!
dphi0 = 0.0         
chi0 = 0.0          
dchi0 = 0.0         

y0 = [phi0, dphi0, chi0, dchi0]

# --- Time Points to Evaluate ---
t_eval = np.linspace(0, t_max, n_points)

# --- Solve the Breathing System ---
sol = solve_ivp(breathing_odes, [0, t_max], y0, t_eval=t_eval, method='RK45')

# --- Extract Solutions ---
phi = sol.y[0]
chi = sol.y[2]
time = sol.t

# --- Plot the Results ---
plt.figure(figsize=(14, 7))
plt.plot(time, phi, label='Breathing Field ϕ(t)')
plt.plot(time, chi, label='Buddy Field χ(t)', linestyle='--')
plt.title("Walter's Breathing Universe: Long-Time Military Test 🌌🛡️")
plt.xlabel('Time')
plt.ylabel('Field Value')
plt.legend()
plt.grid(True)
plt.show()
