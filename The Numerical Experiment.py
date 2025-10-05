import numpy as np
import matplotlib.pyplt as plt 
from matplotlib.animation imoprt FuncAnimation 
import tools as qo 

def heisenberg_4site(J=1.0):
    H = np.zeros((16,16),dtype=complex)
    for i in range(3):  # 4 spins => 3 neighbor pairs
        ops_x = [I2]*4; ops_y = [I2]*4; ops_z = [I2]*4
        ops_x[i], ops_x[i+1] = sx, sx
        ops_y[i], ops_y[i+1] = sy, sy
        ops_z[i], ops_z[i+1] = sz, sz
        H += J*(qo.tensor_product(*ops_x) + qo.tensor_product(*ops_y) + qo.tensor_product(*ops_z))
    return H

def bath_hamiltonian(N=10, omega=0.1):
    a = np.diag(np.sqrt(np.arange(1,N)),1)
    a_dag = a.conj().T
    return omega*(a_dag @ a + 0.5*np.eye(N))

def interaction_hamiltonian(Hb, g=0.2):
    id_spin234 = np.eye(8)
    return g * np.kron(np.kron(sz, id_spin234), Hb)

# ------------------------
# Parameters
# ------------------------
N_bath = 10
omega = 0.1
J = 1.0
g = 0.2
T = 1.0  # bath temperature

# Spin and bath Hamiltonians
Hs = heisenberg_4site(J)
Hb = bath_hamiltonian(N_bath, omega)
Hint = interaction_hamiltonian(Hb, g)

dim_S = Hs.shape[0]
dim_B = Hb.shape[0]
Is = np.eye(dim_S)
Ib = np.eye(dim_B)

# Total Hamiltonian
H_total = np.kron(Hs, Ib) + np.kron(Is, Hb) + Hint

# ------------------------
# Initial state
# ------------------------
up = np.array([1,0],dtype=complex)
down = np.array([0,1],dtype=complex)

psiS = qo.tensor_product(up, down, up, down)  # Néel state |↑↓↑↓>
rhoS0 = np.outer(psiS, psiS.conj())

rhoB0 = np.exp(-Hb/T)
rhoB0 /= np.trace(rhoB0)

rho0 = np.kron(rhoS0, rhoB0)

# ------------------------
# Time evolution (using matrix exponential via eigendecomposition)
# ------------------------
def evolve_rho(rho0, H, t):
    vals, vecs = np.linalg.eigh(H)
    U = vecs @ np.diag(np.exp(-1j*vals*t)) @ vecs.conj().T
    return U @ rho0 @ U.conj().T

# ------------------------
# Observables
# ------------------------
def magnetization(rhoS):
    mags = []
    for i in range(4):
        ops = [I2]*4
        ops[i] = sz
        sz_i = tensor_product(*ops)
        mags.append(np.real(np.trace(rhoS @ sz_i)))
    return mags

# ------------------------
# Simulation loop
# ------------------------
times = np.linspace(0, 10, 50)
dims = [2,2,2,2,N_bath]

magnetizations = []
entropies = []

for t in times:
    rho_t = evolve_rho(rho0, H_total, t)
    rhoS_t = qo.partial_trace(rho_t, dims, trace_out=[4])
    magnetizations.append(magnetization(rhoS_t))
    entropies.append(qo.von_neumann_entropy(rhoS_t))

magnetizations = np.array(magnetizations)
entropies = np.array(entropies)

# ------------------------
# Plot results
# ------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for i in range(4):
    plt.plot(times, magnetizations[:,i], label=f'Spin {i+1}')
plt.xlabel('Time')
plt.ylabel('<σz>')
plt.title('Magnetization decay')
plt.legend()

plt.subplot(1,2,2)
plt.plot(times, entropies)
plt.xlabel('Time')
plt.ylabel('S(ρS)')
plt.title('Entropy growth')

plt.tight_layout()
plt.show()

# ------------------------
# Animation of magnetizations
# ------------------------
plt.ion()  # interactive mode on
fig, ax = plt.subplots()
bars = ax.bar(range(4), magnetizations[0], color='skyblue')
ax.set_ylim(-1,1)
ax.set_xlabel("Spin index")
ax.set_ylabel("<σz>")
ax.set_title("Magnetization evolution over time")

for t_idx in range(len(times)):
    for i, bar in enumerate(bars):
        bar.set_height(magnetizations[t_idx, i])
    ax.set_title(f"Magnetization evolution at t = {times[t_idx]:.2f}")
    plt.pause(0.1)  # pause to update the figure

plt.ioff()
plt.show()
