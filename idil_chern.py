import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import decimal

def H_k(k,q,p):
    kx=k[0]
    ky=k[1]
    # Initialize the matrix
    M = np.zeros((q, q), dtype=complex)

    # Fill in the matrix
    for i in range(q):
        # Main diagonal
        M[i, i] =  2 * np.cos(ky + (2*np.pi)**(1)*(1-p/q) *  (i+1))
    # Upper diagonal
        if i + 1 < q:
            M[i, i + 1] = np.exp(kx*1j)
    # Lower diagonal
        if i - 1 >= 0:
            M[i, i - 1] = np.exp(-kx*1j)

    M[0,q-1]= np.exp(-1j*kx)
    M[q-1,0]= np.exp(1j*kx)  

    if q==2:
        M[0,1]=np.exp(+kx*1j)+np.exp(-kx*1j) 
        M[1,0]=np.exp(+kx*1j)+np.exp(-kx*1j)  
          
    return M

def H(k_vec, dim, p):
    """
    Function to construct the Hamiltonian of a 2DEG in the presence of an applied magnetic field.
    The magnetic field is introduced using Landau's gauge.

    Parameters:
    -----------
    k_vec : array_like
        Vector of length 2 containing the wavevector components kx and ky.
    dim : int
        Dimension of the Hamiltonian, which also corresponds to the number of magnetic flux quanta.

    Returns:
    --------
    Hk : ndarray
        The Hamiltonian matrix in k-space representation.
    """
    Hk = np.zeros((dim, dim), dtype=complex)
    t = 1  # hopping amplitude
    q = dim  # setting q equal to dimension for consistency
    phi = p / q   # Correcting the flux to match the periodicity

    kx, ky = k_vec

    # Diagonal elements
    for i in range(dim):
        Hk[i, i] = -2 * t * np.cos(ky -  (i) * (2 * np.pi)**(1) * phi)

    # Off-diagonal elements
    for i in range(dim - 1):
        Hk[i, i + 1] = -t
        Hk[i + 1, i] = -t  # Ensuring Hermiticity

    # Additional phase for periodic boundary conditions or long-range coupling
    Hk[0, dim - 1] = -t * np.exp(-1j * q * kx)
    Hk[dim - 1, 0] = -t * np.exp(1j * q * kx)  # Ensuring Hermiticity

    return Hk

def build_U(vec1, vec2):
    inner_product = np.dot(vec1, vec2.conj())
    norm = np.linalg.norm(inner_product)
    if norm == 0:
        return 1
    return inner_product / norm

def latF(k_vec, Dk, dim, p):
    """
    Calculate the lattice field F12 as defined in the discrete Berry curvature approach.

    Parameters:
    -----------
    k_vec : array_like
        2D vector of wavevector components (kx, ky) at which to evaluate the Hamiltonian.
    Dk : array_like
        2D vector of the differences in wavevector components used for finite difference approximation.
    dim : int
        Dimension of the Hamiltonian matrix.

    Returns:
    --------
    F12 : ndarray
        Array of lattice field values for each band.
    E_sort : ndarray
        Array of sorted eigenenergies.
    """
    k = k_vec
    E, aux = lg.eig(H_k(k, dim, p))
    idx = E.argsort()
    psi = aux[:, idx]

    k_dx = np.array([k_vec[0] + Dk[0], k_vec[1]], dtype=np.dtype(decimal.Decimal))
    E, aux = lg.eig(H_k(k_dx, dim, p))
    psiDx = aux[:, E.argsort()]

    k_dy = np.array([k_vec[0], k_vec[1] + Dk[1]], dtype=np.dtype(decimal.Decimal))
    E, aux = lg.eig(H_k(k_dy, dim, p))
    psiDy = aux[:, E.argsort()]

    k_dxdy = np.array([k_vec[0] + Dk[0], k_vec[1] + Dk[1]], dtype=np.dtype(decimal.Decimal))
    E, aux = lg.eig(H_k(k_dxdy, dim, p))
    psiDxDy = aux[:, E.argsort()]

    U1x = np.array([build_U(psi[:, i], psiDx[:, i]) for i in range(dim)])
    U2y = np.array([build_U(psi[:, i], psiDy[:, i]) for i in range(dim)])
    U1y = np.array([build_U(psiDy[:, i], psiDxDy[:, i]) for i in range(dim)])
    U2x = np.array([build_U(psiDx[:, i], psiDxDy[:, i]) for i in range(dim)])

    F12 = np.log( U1x * U2x * 1./U1y * 1./U2y)

    return F12, E[idx]

x_res = 27
y_res = 27
q = 3
p = 1
Nd = q

Dx = (2 * np.pi / q) / x_res
Dy = (2 * np.pi) / y_res
Dk = np.array([Dx, Dy], dtype=np.dtype(decimal.Decimal))

LF = np.zeros((Nd), dtype=complex)
LF_arr = np.zeros((Nd, x_res, y_res),dtype=np.dtype(decimal.Decimal))
sumN = np.zeros((Nd), dtype=complex)
E_k = np.zeros((Nd), dtype=complex)
chernN = np.zeros((Nd), dtype=complex)

for ix in range(x_res):
    kx = ix * Dx
    for iy in range(y_res):
        ky = iy * Dy
        k_vec = np.array([kx, ky], dtype=np.dtype(decimal.Decimal))
        LF, E_k = latF(k_vec, Dk, Nd, p)
        sumN += LF
        LF_arr[:, ix, iy] = -LF.imag / (2 * np.pi)

chernN = sumN.imag / (2 * np.pi)
print("Chern number associated with each band: ", chernN)
print("Rounded chern number associated with each band: ", np.round(chernN).astype(int))
print("Sum of rounded chern numbers: ", sum(np.round(chernN).astype(int)))

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

kx = np.linspace(0, 2 * np.pi / q, x_res)
ky = np.linspace(0, 2 * np.pi, y_res)

kx, ky = np.meshgrid(ky, kx)

surf = ax.plot_wireframe(ky, kx, LF_arr[1, :, :], rstride=1, cstride=1, color='0.4')

ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_xticks([0.0, np.pi / q, 2 * np.pi / q])
ax.set_xticklabels([r'$0$', r'$\pi/3$', r'$2\pi/3$'], fontsize=16)
ax.set_xlim(0, 2 * np.pi / q)

ax.set_ylabel(r'$k_y$', fontsize=18)
ax.set_yticks([0.0, np.pi, 2 * np.pi])
ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=16)
ax.set_ylim(0, 2 * np.pi)

ax.set_zlabel(r'$i\tilde{F}_{12}$', fontsize=18)
ax.set_zlim(np.min(LF_arr[1, :, :]),np.max(LF_arr[1, :, :]))

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 1.5, 1, 1]))

pl.show()
