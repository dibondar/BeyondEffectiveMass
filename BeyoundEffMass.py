import numpy as np
from scipy.constants import c, mu_0, epsilon_0, e, hbar
import matplotlib.pyplot as plt


class CBeyondEffMass:
    """
    Implementation of the numerical method from
    "Beyond the effective mass approximation: A predictive theory of the nonlinear optical response of conduction electrons"
    https://doi.org/10.1103/PhysRevB.95.125201

    /================================================================================================|
    / Region 1 (vacuum)                       |   Region 2 (material)         | Region 3 (vacuum)    |
    ==================================================================================================
    / characterized by                        |                               |                      |
    / the coordinate step size Δx1            |         Δx2                   |          Δx3         |
    / the number of coordinate points  n1     |         n2                    |          n3          |
    /                                         |                               |                      |
    /  the electric and magentic fields       |                               |                      |
    /      Ez1, Hy1                           |     Ez2, Hy2,                 |        Ez3, Hy3      |
    /                                         |    Dz2 (displacement field)   |                      |
    /================================================================================================|

    Notes: n1 will be set as Ez_init.size. Hy stand for H_y^\#.
    """
    def __init__(self, *, Δx1:float, Δt:float, Ez_init:np.ndarray, diff_E,
                 n:float = 0, γ:float = 0, epsilon_infty:float = 1,
                 Δx2:float = None, Δx3:float = None, N2:int = None, N3:int = None,):
        """
        :param n: electron density
        :param γ: the scattering rate in the Drude model
        :param epsilon_infty: the background dielectric constant
        :param diff_E: The derivative of the conduction band
        :param x:
        :param Δt:
        :param Ez1_init: The array specifying the initial condition for the electric field localized in region 1.
        :return: self
        """
        Δx3 = Δx3 if Δx3 else Δx1
        Δx2 = Δx2 if Δx2 else Δx1

        N3 = N3 if N3 else Ez_init.size
        N2 = N2 if N2 else N3

        self.Δx1 = Δx1
        self.Δx2 = Δx2
        self.Δx3 = Δx3

        self.Δt = Δt

        ################################################################################################################
        #
        #       Initializing EM Fields
        #
        ################################################################################################################

        # numerical constants for the Yee method
        # Please note that we halved the time step for half-time propagation of the Maxwell equation
        self.k1 = 0.5 * Δt / (mu_0 * Δx1)
        self.k2 = 0.5 * Δt / (mu_0 * Δx2)
        self.k3 = 0.5 * Δt / (mu_0 * Δx3)

        self.c1 = 0.5 * Δt / (epsilon_0 * Δx1)
        self.c2 = 0.5 * Δt / Δx2
        self.c3 = 0.5 * Δt / (epsilon_0 * Δx3)

        # Initializing electric fields
        self.Ez1 = Ez_init.copy()
        self.Ez2 = np.zeros(N2)
        self.Ez3 = np.zeros(N3)

        self.Dz2 = np.zeros(N2)

        # Initializing magnetic fields
        self.Hy1 = -self.Ez1 / (c * mu_0)

        # preparing for the propagation doing a half step for Hy1
        self.Hy1[1:-1] += (self.Hy1[2:] - self.Hy1[:-2]) / 4 + Δt / (4 * mu_0 * Δx1) * (self.Ez1[2:] - self.Ez1[:-2])
        self.Hy1[0] += self.Hy1[1] / 4 + Δt / (4 * mu_0 * Δx1) * self.Ez1[1]
        self.Hy1[-1] += -self.Hy1[-2] / 4 - Δt / (4 * mu_0 * Δx1) * self.Ez1[-2]

        self.Hy2 = np.zeros(N2)
        self.Hy3 = np.zeros(N3)

        # Initialize the coordinate grids for visualization
        self.x1 = Δx1 * np.arange(self.Ez1.size)
        self.x2 = self.x1[-1] + Δx1 + Δx2 * np.arange(N2)
        self.x3 = self.x2[-1] + Δx2 + Δx3 * np.arange(N3)

        ################################################################################################################
        #
        #      Initializing Matter
        #
        ################################################################################################################

        self.diff_E = diff_E
        self.n = n
        self.γ = γ
        self.epsilon_infty = epsilon_infty

        # wave vectors for each spatial point in region 2
        self.k = np.zeros(N2)

        # nonlinear polarizability for each spatial point in region 2
        self.PNL = np.zeros(N2)

        # Coefficients
        self.coeff_PNL = -n * e / hbar
        self.coeff_E = -e / (hbar * epsilon_0 * epsilon_infty)

    def __half_step_maxwell_propagation(self):
        """
        Perform a half time-sep propagation for the Maxwell equation via the Yee method
        :return:
        """
        # aliases
        Ez1 = self.Ez1
        Ez2 = self.Ez2
        Ez3 = self.Ez3

        Hy1 = self.Hy1
        Hy2 = self.Hy2
        Hy3 = self.Hy3

        Dz2 = self.Dz2

        ################################################################################################################
        #
        #       Region 1
        #
        ################################################################################################################

        Hy1[:-1] += self.k1 * (Ez1[1:] - Ez1[:-1])

        # the boundary condition
        Hy1[-1] += self.k1 * (Ez2[0] - Ez1[-1])

        Ez1[1:] += self.c1 * (Hy1[1:] - Hy1[:-1])

        # the boundary condition
        Ez1[0] += self.c1 * Hy1[0]

        ################################################################################################################
        #
        #       Region 2
        #
        ################################################################################################################

        Hy2[:-1] += self.k2 * (Ez2[1:] - Ez2[:-1])

        # the boundary condition
        Hy2[-1] += self.k2 * (Ez3[0] - Ez2[-1])

        Dz2[1:] += self.c2 * (Hy2[1:] - Hy2[:-1])

        # the boundary condition
        Dz2[0] += epsilon_0 * self.c1 * (Hy2[0] - Hy1[-1])

        ################################################################################################################
        #
        #       Region 3
        #
        ################################################################################################################

        Hy3[:-1] += self.k3 * (Ez3[1:] - Ez3[:-1])

        # the boundary condition
        Hy3[-1] += -self.k3 * Ez3[-1]

        Ez3[1:] += self.c3 * (Hy3[1:] - Hy3[:-1])

        # the boundary condition
        Ez3[0] += self.c2 / epsilon_0 * (Hy3[0] - Hy2[-1])

    def __matter_single_step_propagation(self):
        """
        Solving differential equations for PNL and k using RK4
        :return:
        """
        # Aliases
        k = self.k
        PNL = self.PNL
        Dz2 = self.Dz2
        Δt = self.Δt
        γ = self.γ
        coeff_E = self.coeff_E
        coeff_PNL = self.coeff_PNL

        # Runge–Kutta Propagation https://dlmf.nist.gov/3.7#v
        PNL1 = Δt * coeff_PNL * self.diff_E(k)
        K1 = Δt * (-γ * k + coeff_E * (Dz2 - PNL))

        PNL2 = Δt * coeff_PNL * self.diff_E(k + 0.5 * K1)
        K2 = Δt * (-γ * (k + 0.5 * K1) + coeff_E * (Dz2 - PNL - 0.5 * PNL1))

        PNL3 = Δt * coeff_PNL * self.diff_E(k + 0.5 * K2)
        K3 = Δt * (-γ * (k + 0.5 * K2) + coeff_E * (Dz2 - PNL - 0.5 * PNL2))

        PNL4 = Δt * coeff_PNL * self.diff_E(k + K3)
        K4 = Δt * (-γ * (k + K3) + coeff_E * (Dz2 - PNL - PNL3))

        k += (K1 + 2 * K2 + 2 * K3 + K4) / 6
        PNL += (PNL1 + 2 * PNL2 + 2 * PNL3 + PNL4) / 6

        # Updating the electric field in the bulk
        self.Ez2 = (Dz2 - PNL) / (epsilon_0 * self.epsilon_infty)

    def single_step_propagation(self):
        """
        Propagate the entire EM-Matter system via the single step propagation
        :return:
        """
        self.__half_step_maxwell_propagation()
        self.__matter_single_step_propagation()
        self.__half_step_maxwell_propagation()

    def plot_Ez(self, bulk_color="b"):
        """
        Plot the current electric field
        :return:
        """
        plt.plot(self.x1, self.Ez1 / 1e5, "y", label="vacuum")
        plt.plot(self.x2, self.Ez2 / 1e5,  "b", label="bulk")
        plt.plot(self.x3, self.Ez3 / 1e5, "y")
        plt.xlabel("$x$ (m)")
        plt.ylabel("Electric field $E_z$ (kV/cm)")
        plt.axvspan(self.x2[0], self.x2[-1], color=bulk_color, alpha=0.2)