import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import torch
import torch.nn as nn

class System:
    """System of dynamics with mass-spring-damper modeling."""
    
    def __init__(self, M: np.ndarray, C: np.ndarray, K: np.ndarray, dof_names: list[str] | None = None):
        self._M = M
        self._C = C
        self._K = K
        self.dof_names = dof_names
        self.A, self.B = self.state_space()

    @property
    def M(self):
        return self._M
    @M.setter
    def M(self, value):
        self._M = value
        self.A, self.B = self.state_space()

    @property
    def C(self):
        return self._C
    @C.setter
    def C(self, value):
        self._C = value
        self.A, self.B = self.state_space()

    @property
    def K(self):
        return self._K
    @K.setter
    def K(self, value):
        self._K = value
        self.A, self.B = self.state_space()

    def state_space(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert M-C-K matrices to state-space form."""
        n = self._M.shape[0]
        
        try:
            np.linalg.cholesky(self._M)
        except np.linalg.LinAlgError:
            raise ValueError("Mass matrix M must be positive definite (invertible).")
        
        A = np.zeros((2*n, 2*n))
        B = np.zeros((2*n, n))
        
        invM = np.linalg.inv(self._M)
        
        # Structure : x = [q, vq]
        # dq/dt = vq
        A[:n, n:] = np.eye(n)
        # dvq/dt = -M^-1 * (K*q + C*vq)
        A[n:, :n] = -invM @ self.K
        A[n:, n:] = -invM @ self.C
        
        # dvq/dt = M^-1 * F(t)
        B[n:, :] = invM
        
        return A, B

    def simulate(self, x0, t, u):
        """Simulate linear system: dx/dt = A*x + B*u(t).

        Args:
            x0: Initial state at t[0]
            t: Time array for output
            u: Input function u(t)

        Returns:
            State trajectory (n_states, n_timepoints) array.
        """
        def deriv(t_curr, x_curr):
            return self.A @ x_curr + self.B @ u(t_curr)
        
        sol = solve_ivp(deriv, (t[0], t[-1]), x0, t_eval=t, method='RK45')
        return sol.y.T
    
    def eigenvalues(self):
        return np.linalg.eigvals(self.A)
    
    def res_freq(self):
        eigvals = self.eigenvalues()
        imag_parts = np.abs(np.imag(eigvals))
        frequencies = np.unique(imag_parts / (2 * np.pi))
        return frequencies[frequencies > 0]
    
    def plot_response(self, x0, t, u):
        response = self.simulate(x0, t, u)
        
        # On ne trace que les positions (première moitié des colonnes)
        num_dofs = self.M.shape[0]
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        
        for i in range(num_dofs):
            label = f'{self.dof_names[i]}' if self.dof_names else f'x{i+1}'
            ax1.plot(t, response[:, i], label=label)
            
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Displacement (m)')
        ax1.set_title('System Response')
        ax1.legend(loc='lower right', ncol=2)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def animate_response(response, t, f, kr, xr_offset=25, xr_max=10, xr_target_offset=5, fps=30):
    """Create animation of mass-spring-damper system response.
    
    Args:
        response: Simulated state responses [time_steps, 2*n_dofs]
        t: Time array
        f: Force input function
        kr: Spring stiffness coefficient
        xr_offset: Offset for right spring position (cm)
        xr_max: Maximum displacement range for animation (cm)
        xr_target_offset: Target position offset (cm)
        fps: Animation frames per second
        decimation_factor: Optional override for time step decimation
    
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    decimation_factor = int(len(t) / (t[-1] * fps))
    time_decimated = t[::decimation_factor]
    xr = response[:,0][::decimation_factor]*100
    xb = response[:,1][::decimation_factor]*100
    f_decimated = [f(ti) for ti in time_decimated]
    x_input = np.array(f_decimated) / kr * 100
    xr_target = xr_offset + xr_max + xr_target_offset
    max_xr = max(response[:,0]*100 + response[:,1]*100) + xr_offset
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.tight_layout()

    ax.plot([0, 0], [-5, 5], color='black', linewidth=2, zorder=1)
    ax.plot([xr_offset+xr_max, xr_offset+xr_max], [-5, 5], color='black', linewidth=2, linestyle='--', zorder=1)
    ax.plot([xr_offset-xr_max, xr_offset-xr_max], [-5, 5], color='black', linewidth=2, linestyle='--', zorder=1)
    ax.plot([xr_target, xr_target], [-5, 5], color='red', linewidth=2, zorder=1)
    ax.annotate('xr target', xy=(xr_target, -5.5), fontsize=12, color='red', ha='center', va='top')
    ax.annotate('xr max', xy=(xr_offset+xr_max, 5.5), fontsize=12, color='black', ha='center')
    ax.annotate('xr min', xy=(xr_offset-xr_max, 5.5), fontsize=12, color='black', ha='center')

    xinput = ax.plot([xr_offset, xr_offset], [-5, 5], color='green', linewidth=2, linestyle='--', zorder=1)
    xinput_annotate = ax.annotate('Input Position', xy=(xr_offset, -6.5), fontsize=12, color='green', ha='center', va='top')

    m1 = patches.Rectangle((-2+xb[0], -3), 4, 6, fc=(0.8, 0.8, 0.8, 0.7), ec='black', linewidth=2, zorder=3)
    ax.add_patch(m1)
    centre_m1 = ax.plot([xb[0]], [0], marker='o', markersize=5, color='blue', zorder=4)

    m2 = patches.Circle((xr_offset+xr[0], 0), 2, fc=(0.8, 0.8, 0.8, 0.7), ec='black', linewidth=2, zorder=4)
    ax.add_patch(m2)
    centre_m2 = ax.plot([xr_offset+xr[0]], [0], marker='o', markersize=5, color='red', zorder=5)

    spring1_x = np.linspace(0, xb[0], 10)
    spring1_y = np.zeros_like(spring1_x)
    spring1_y[1:-1:2] = 1
    spring1_y[2:-1:2] = -1
    spring1 = ax.plot(spring1_x, spring1_y, color='blue', linewidth=2, zorder=2)

    spring2_x = np.linspace(2+xb[0], xr_offset+xr[0], 15)
    spring2_y = np.zeros_like(spring2_x)
    spring2_y[1:-1:2] = 1
    spring2_y[2:-1:2] = -1
    spring2 = ax.plot(spring2_x, spring2_y, color='red', linewidth=2, zorder=2)

    anot_str = f't = {time_decimated[0]:.2f} s\nxb = {xb[0]:.2f} cm\nxr = {xr[0]:.2f} cm'
    anotation = ax.text(0.02, 0.95, anot_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    anot2_str = f'xr target = {xr_target:.2f} cm\nMax xr reached = {max_xr:.2f} cm'
    ax.text(0.02, 0.05, anot2_str, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(-10, 45)
    ax.set_ylim(-10, 10)
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Position (cm)')
    ax.set_title('Mass-Spring-Damper System Animation')

    def update(frame):
        x_m1 = float(-2 + xb[frame])
        m1.set_xy((x_m1, -3))
        centre_m1[0].set_data([xb[frame]], [0])

        x_m2 = float(xr_offset + xr[frame] + xb[frame])
        m2.set_center((x_m2, 0))
        centre_m2[0].set_data([x_m2], [0])

        spring1_x = np.linspace(0, x_m1+2, 10)
        spring1_y = np.zeros_like(spring1_x)
        spring1_y[1:-1:2] = 1
        spring1_y[2:-1:2] = -1
        spring1[0].set_data(spring1_x, spring1_y)

        spring2_x = np.linspace(x_m1+4, x_m2, 15)
        spring2_y = np.zeros_like(spring2_x)
        spring2_y[1:-1:2] = 1
        spring2_y[2:-1:2] = -1
        spring2[0].set_data(spring2_x, spring2_y)

        xinput_annotate.set_position((x_input[frame] + xr_offset + xb[frame], -6.5))
        xinput[0].set_data([x_input[frame] + xr_offset + xb[frame], x_input[frame] + xr_offset + xb[frame]], [-5, 5])

        anot_str = f't = {time_decimated[frame]:.2f} s\nxb = {xb[frame]:.2f} cm\nxr = {xr[frame]:.2f} cm'
        anotation.set_text(anot_str)
        return m1, spring1[0], centre_m1[0], anotation

    ani = animation.FuncAnimation(fig, update, frames=len(time_decimated), interval=1000/fps, blit=True)
    plt.close(fig)
    return ani

def chirp_hold(t, f0, f1, T, amplitude=1.0):
    """Generate FM chirp signal with frequency hold.

    Args:
        t: time float (s)
        f0: Start frequency (Hz)
        f1: End frequency (Hz)  
        T: Total sweep + hold duration (s)
        amplitude: Signal amplitude

    Returns:
        tuple: (signal waveform, hold_frequency)

    Note: Frequency ramps from f0→f1 during first T/2 s, then holds constant.
    """
    if t <= T / 2:
        duration = T / 2
        target_phase = np.pi / 2

        if np.isclose(f1, f0):
            t1 = duration
            beta = 0.0
        else:
            n_cycles = np.ceil((duration * (f0 + f1) - target_phase / np.pi) / 2.0)
            t1 = 2.0 * (f1 * duration - n_cycles - target_phase / (2.0 * np.pi)) / (f1 - f0)
            t1 = np.clip(t1, 0.0, duration)
            beta = (f1 - f0) / t1 if t1 > 0 else 0.0

        if t <= t1:
            phase = 2 * np.pi * (f0 * t + 0.5 * beta * t**2)
            frequency = f0 + beta * t
        else:
            phase = 2 * np.pi * (f0 * t1 + 0.5 * beta * t1**2 + f1 * (t - t1))
            frequency = f1
        return amplitude * np.sin(phase), frequency
    else:
        return amplitude, 0.0
    

def evaluate_performance(system, T, f1, amplitude_cm, kr, xr_offset=25, xr_max=10, xr_target_offset=5, plot_response=False):
    """Evaluate robot performance metrics from system simulation.

    Generates a chirp control input and simulates system response to compute
    maximum reached position relative to target.

    Args:
        system: robot or similar dynamical system instance
        T (float): Total simulation duration (s)
        f1 (float): Target frequency for chirp sweep (Hz)
        amplitude_cm (float): Chirp amplitude (cm units)
        kr (float): Robot spring stiffness (N/m)
        xr_offset, xr_max, xr_target_offset: Position tracking targets

    Returns:
        tuple: (error_metric, animation) where:
            - error_metric: max_reached - target position difference (cm)
            - animation: Animation object or None if plot_response=False

    Note:
        Chirp frequency signal starts at 0 Hz and sweeps to f1 during T.
        System simulates linear dynamics: dx/dt = A*x + B*u(t). Maximum
        reached position combines both DoFs (xr + xb).
    """

    x0 = np.zeros(2 * system.M.shape[0])
    time = np.linspace(0, T, int(T * 200))
    f = lambda t: chirp_hold(t, 0, f1, T, amplitude_cm*kr/100)[0]
    u = lambda t: np.array([f(t), -f(t)])
    response = system.simulate(x0, time, u)
    max_reached = max(response[:,0] + response[:,1]) * 100
    if plot_response:
        ani = animate_response(response, time, f, kr, xr_offset=xr_offset, xr_max=xr_max, xr_target_offset=xr_target_offset)
    else:
        ani = None
    return max_reached, ani


    
def generate_training_data(system_config_list, n_samples=None, plot_samples=False):
    """Generate training dataset for the NN model.
    
    Args:
        system_config_list: List of dicts with keys ['hr', 'hb', 'kr', 'kb', 'amplitude_cm', 'f1', 'T', 'mr']
        n_samples: Override number of samples (None uses provided list)
        plot_samples: Whether to plot the generated samples
    Returns:
        X (torch.FloatTensor): Input features for NN
        y (torch.FloatTensor): Target performance metric (outreach)
        anims (list): List of animations for each sample (or None if not plotted)
    """
    inputs = []
    targets = []
    anims = []
    
    if n_samples is not None and len(system_config_list) == 1:
        samples = [[system_config_list[0]] * n_samples]
    else:
        samples = system_config_list
        
    mb = 10  # Fixed mass of the base for all samples
    for cfg in samples:
        mr = cfg['mr']
        kr = cfg['kr']
        kb = cfg['kb']
        hr = cfg['hr']
        hb = cfg['hb']
        amplitude_cm = cfg['amplitude']
        f1 = cfg['f1']
        period = cfg['T']
        
        cr = 2 * hr * np.sqrt(kr * mr)
        cb = 2 * hb * np.sqrt(kb * mb)
        
        M = np.array([[mr, mr], [0, mb]])
        C = np.array([[cr, 0], [-cr, cb]])
        K = np.array([[kr, 0], [-kr, kb]])
        
        robotTrain = System(M, C, K, ['xr', 'xb'])
        
        # Normalize input features for the neural network
        input_features = np.array([
            hr/0.05,
            hb/0.1,
            kr/10000.0,
            kb/3000.0,
            mr/4.0,
            amplitude_cm/10.0,
            f1/2.5,
            period/10.0])
        
        # Evaluate performance (outreach)
        try:
            outreach, ani = evaluate_performance(robotTrain, period, f1, amplitude_cm, kr, plot_response=plot_samples)
            inputs.append(input_features)
            targets.append(outreach)
            anims.append(ani)

        except Exception as e:
            print(f"Error evaluating performance for config {cfg}: {e}")
            continue
    
    inputs = torch.FloatTensor(np.array(inputs))
    targets = torch.FloatTensor(np.array(targets)).reshape(-1, 1)
    
    return inputs, targets, anims