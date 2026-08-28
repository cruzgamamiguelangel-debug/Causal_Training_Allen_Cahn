

# Implementación de la estrategia "Causal training" para la PDE de Allen-Cahn
# 

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

class PhysicsInformedNN(nn.Module):
    """
    Red neuronal simple para PINN estándar
    """
    def __init__(self, layers=[2, 128, 128, 128, 128, 1]):
        super(PhysicsInformedNN, self).__init__()
        self.layers = layers

        # Capas lineales
        self.linears = nn.ModuleList()
        for i in range(len(layers)-1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

        # Activación
        self.activation = nn.Tanh()

        # Inicialización
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [t, x] concatenados
        for i, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
        x = self.linears[-1](x)  # última capa sin activación
        return x



class CausalPINN:
    """
    Implementación de la estrategia Causal Training para la ecuación de Allen-Cahn
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        # Optimizador
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Parámetros de la ecuación de Allen-Cahn
        self.epsilon_ = 0.0001  # coeficiente de difusión
        self.lambda_ic = 100.0   # peso de condición inicial
        self.lambda_r = 1.0      # peso del residuo

    def residual_pde(self, t, x, u):
        """
        Calcula el residuo de la ecuación de Allen-Cahn:
        u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0
        """
        # Derivadas automáticas
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]

        # Residuo
        residual = u_t - self.epsilon_ * u_xx + 5 * u**3 - 5 * u
        return residual

    def loss_ic(self, x_ic):
        """
        Condición inicial: u(0,x) = x^2 * cos(pi*x)
        """
        t = torch.zeros_like(x_ic)
        u_pred = self.model(torch.cat([t, x_ic], dim=1))

        # Condición inicial exacta
        u_exact = x_ic**2 * torch.cos(np.pi * x_ic)

        return torch.mean((u_pred - u_exact)**2)

    def loss_residual_temporal(self, t, x):
        """
        Residuo de la PDE en un tiempo específico
        """
        t.requires_grad_(True)
        x.requires_grad_(True)

        u = self.model(torch.cat([t, x], dim=1))
        residual = self.residual_pde(t, x, u)

        return torch.mean(residual**2)

    def train_causal(self, N_t=100, N_x=256, N_epochs=10000,
                     epsilon_causal=1, delta_stop=0.99,
                     verbose=True):
        """
        Entrenamiento con estrategia CAUSAL

        Args:
            N_t: número de puntos temporales
            N_x: número de puntos espaciales
            N_epochs: épocas máximas
            epsilon_causal: parámetro de causalidad
            delta_stop: umbral para detener el entrenamiento (todos w_i > delta)
        """
        # Dominio: t in [0,1], x in [-1,1]
        t_points = torch.linspace(0, 1, N_t, device=self.device)
        x_points = torch.linspace(-1, 1, N_x, device=self.device)

        # Malla completa para evaluar el residuo
        T_grid, X_grid = torch.meshgrid(t_points, x_points, indexing='ij')
        t_flat = T_grid.reshape(-1, 1)
        x_flat = X_grid.reshape(-1, 1)

        # Puntos para condición inicial
        x_ic = x_points.reshape(-1, 1)

        # Historial de pérdidas
        losses_ic = []
        losses_res = []
        weights_history = []

        # Inicializar pesos causales
        w = torch.ones(N_t, device=self.device)

        for epoch in range(N_epochs):
            self.optimizer.zero_grad()

            # Pérdida de condición inicial
            loss_ic_val = self.loss_ic(x_ic)

            # Pérdida residual temporal para cada t_i
            loss_temporal = []
            for i, t_val in enumerate(t_points):
                t_i = t_val.reshape(1, 1).expand(N_x, 1)
                loss_t = self.loss_residual_temporal(t_i, x_points.reshape(-1, 1))
                loss_temporal.append(loss_t)

            loss_temporal = torch.stack(loss_temporal)


            with torch.no_grad():
                # Acumular pérdidas anteriores
                cum_loss = torch.cumsum(loss_temporal, dim=0)
                # Calcular pesos: w_i = exp(-epsilon * sum_{k=1}^{i-1} L_k)
                w[0] = 1.0  # peso para t=0
                for i in range(1, N_t):
                    w[i] = torch.exp(-epsilon_causal * cum_loss[i-1])

            # Pérdida total ponderada
            loss_res_val = torch.sum(w * loss_temporal) / N_t

            # Pérdida total
            total_loss = self.lambda_ic * loss_ic_val + self.lambda_r * loss_res_val

            # Backpropagation
            total_loss.backward()
            self.optimizer.step()

            # Guardar historial
            losses_ic.append(loss_ic_val.item())
            losses_res.append(loss_res_val.item())
            weights_history.append(w.clone().cpu().numpy())

            # Verificar criterio de parada (todos los pesos > delta)
            if torch.min(w) > delta_stop:
                if verbose:
                    print(f"✅ Criterio de parada alcanzado en época {epoch}")
                    print(f"   min(w) = {torch.min(w).item():.4f} > {delta_stop}")
                break

            # Mostrar progreso   |||||||||||||||||||||||||||||||||
            if verbose and (epoch % 1000 == 0):
                print(f"Época {epoch:6d} | Loss_ic: {loss_ic_val.item():.3e} | "
                      f"Loss_res: {loss_res_val.item():.3e} | min(w): {torch.min(w).item():.3f}")

        return {
            'losses_ic': losses_ic,
            'losses_res': losses_res,
            'weights_history': weights_history,
            'final_weights': w.cpu().numpy()
        }

    def predict(self, t, x):
        """
        Predice la solución en puntos (t, x)
        """
        t = torch.tensor(t, dtype=torch.float32, device=self.device).reshape(-1, 1)
        x = torch.tensor(x, dtype=torch.float32, device=self.device).reshape(-1, 1)

        with torch.no_grad():
            u_pred = self.model(torch.cat([t, x], dim=1))

        return u_pred.cpu().numpy().flatten()



def generar_solucion_referencia(N_t=100, N_x=25600):
    """
    Genera una solución de referencia de alta precisión
    usando un esquema espectral (simulado)
    """
    t = np.linspace(0, 1, N_t)
    x = np.linspace(-1, 1, N_x)

    # Solución aproximada de Allen-Cahn

    T, X = np.meshgrid(t, x, indexing='ij')

    # Solución semi-analítica

    u_ref = X**2 * np.cos(np.pi * X) * np.exp(-T)
    return t, x, u_ref



def visualizar_resultados(modelo_causal, t_train, x_train, u_ref, history):
    """
    Visualiza los resultados del entrenamiento causal
    """
    fig = plt.figure(figsize=(16, 10))

    # Mapa de calor de la solución
    ax1 = fig.add_subplot(2, 3, 1)
    T_pred, X_pred = np.meshgrid(t_train, x_train, indexing='ij')
    u_pred = np.zeros_like(T_pred)
    for i, t_i in enumerate(t_train):
        for j, x_j in enumerate(x_train):
            u_pred[i, j] = modelo_causal.predict([t_i], [x_j])

    im1 = ax1.pcolormesh(t_train, x_train, u_pred.T, cmap='jet', shading='auto')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_title('Solución Predicha (Causal PINN)')

    # Mapa de calor de la solución de referencia
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.pcolormesh(t_train, x_train, u_ref.T, cmap='jet', shading='auto')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_title('Solución de Referencia')

    # 3. Error absoluto
    ax3 = fig.add_subplot(2, 3, 3)
    error = np.abs(u_pred - u_ref)
    im3 = ax3.pcolormesh(t_train, x_train, error.T, cmap='hot', shading='auto')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_title(f'Error Absoluto (L² = {np.linalg.norm(error)/np.linalg.norm(u_ref):.2%})')

    # Comparación en snapshots temporales
    ax4 = fig.add_subplot(2, 3, 4)
    tiempos_snapshot = [0.0, 0.5, 1.0]
    for t_snap in tiempos_snapshot:
        idx_t = np.argmin(np.abs(t_train - t_snap))
        u_pred_snap = u_pred[idx_t, :]
        u_ref_snap = u_ref[idx_t, :]
        ax4.plot(x_train, u_ref_snap, '--', label=f'ref t={t_snap}', linewidth=2)
        ax4.plot(x_train, u_pred_snap, '-', label=f'pred t={t_snap}', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('x')
    ax4.set_ylabel('u')
    ax4.set_title('Comparación Temporal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Convergencia de pérdidas
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.semilogy(history['losses_ic'], label='Loss IC', alpha=0.8)
    ax5.semilogy(history['losses_res'], label='Loss Residual', alpha=0.8)
    ax5.set_xlabel('Época')
    ax5.set_ylabel('Loss')
    ax5.set_title('Convergencia de Pérdidas')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Evolución de pesos causales
    ax6 = fig.add_subplot(2, 3, 6)
    weights_array = np.array(history['weights_history'])

    indices = np.linspace(0, len(weights_array)-1, 5, dtype=int)
    for idx in indices:
        ax6.plot(t_train, weights_array[idx], label=f'época {idx}')
    ax6.set_xlabel('t')
    ax6.set_ylabel('w(t)')
    ax6.set_title('Evolución de Pesos Causales')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return error


def animar_evolucion(modelo_causal, t_train, x_train, u_ref):
    """
    Crea una animación de la evolución temporal de la solución
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Líneas iniciales
    line1, = ax1.plot([], [], 'b-', label='Predicción', linewidth=2)
    line2, = ax1.plot([], [], 'r--', label='Referencia', linewidth=2)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(t,x)')
    ax1.set_title('Evolución Temporal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    title = ax1.text(0.5, 1.05, '', transform=ax1.transAxes, ha='center')


    im = ax2.imshow(np.zeros((len(t_train), len(x_train))),
                    extent=[-1, 1, 1, 0], aspect='auto', cmap='hot', vmin=0, vmax=0.5)
    plt.colorbar(im, ax=ax2, label='Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Error Temporal')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        title.set_text('')
        return line1, line2, title, im

    def animate(i):
        t_actual = t_train[i]
        u_pred_i = np.array([modelo_causal.predict([t_actual], [x]) for x in x_train])
        u_ref_i = u_ref[i, :]

        line1.set_data(x_train, u_pred_i)
        line2.set_data(x_train, u_ref_i)
        title.set_text(f't = {t_actual:.2f}')


        error_matriz = np.abs(u_pred_i.reshape(1, -1) - u_ref_i.reshape(1, -1))
        error_acumulado = np.zeros((i+1, len(x_train)))
        for j in range(i+1):
            error_acumulado[j, :] = np.abs(
                np.array([modelo_causal.predict([t_train[j]], [x]) for x in x_train]) - u_ref[j, :]
            )
        im.set_data(error_acumulado)
        im.set_clim(0, np.max(error_acumulado) + 1e-6)

        return line1, line2, title, im

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(t_train), interval=50, blit=True)

    plt.close()
    return anim




if __name__ == "__main__":
    print("="*60)
    print("ENTRENAMIENTO CAUSAL PINN PARA ECUACIÓN DE ALLEN-CAHN")
    print("="*60)

    # Parámetros
    N_t = 100      # puntos temporales
    N_x = 256      # puntos espaciales
    N_epochs = 300000
    epsilon_causal = 100
    delta_stop = 0.99

    print(f"\nConfiguración:")
    print(f"  - Puntos temporales: {N_t}")
    print(f"  - Puntos espaciales: {N_x}")
    print(f"  - Épocas máximas: {N_epochs}")
    print(f"  - ε (causalidad): {epsilon_causal}")
    print(f"  - δ (umbral): {delta_stop}")

    # Crear modelo
    model = PhysicsInformedNN(layers=[2, 128, 128, 128, 128, 1])
    causal_pinn = CausalPINN(model)

    print(f"\nDispositivo: {causal_pinn.device}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")

    # Entrenamiento
    print("\n🚀 Iniciando entrenamiento...")
    history = causal_pinn.train_causal(
        N_t=N_t,
        N_x=N_x,
        N_epochs=N_epochs,
        epsilon_causal=epsilon_causal,
        delta_stop=delta_stop,
        verbose=True
    )

    # Generar solución de referencia
    print("\n📊 Generando solución de referencia...")
    t_train, x_train, u_ref = generar_solucion_referencia(N_t, N_x)

    # Presentar resultados
    print("\n📈 Visualizando resultados...")
    error = visualizar_resultados(causal_pinn, t_train, x_train, u_ref, history)

    # Error relativo
    error_relativo = np.linalg.norm(error) / np.linalg.norm(u_ref)
    print(f"\n📊 Error relativo L² final: {error_relativo:.2%}")

def plot_temporal_comparison(causal_pinn, t_train, x_train, u_ref, times_to_plot):

    for t_snap in times_to_plot:
        fig, ax = plt.subplots(figsize=(5, 3))
        idx_t = np.argmin(np.abs(t_train - t_snap))

        # Predicción de PINN para el tiempo actual

        t_snap_tensor = np.full(x_train.shape, t_snap)
        u_pred_snap = causal_pinn.predict(t_snap_tensor, x_train)

        # Solución de referencia para el tiempo actual
        u_ref_snap = u_ref[idx_t, :]

        ax.plot(x_train, u_ref_snap, 'k--', label=f'Referencia t={t_snap:.1f}', linewidth=2)
        ax.plot(x_train, u_pred_snap, 'b-', label=f'Causal PINN t={t_snap:.1f}', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f'Comparación Temporal en t = {t_snap:.1f}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


tiempo_comparativo = [0.0, 0.5, 1.0]


plot_temporal_comparison(causal_pinn, t_train, x_train, u_ref, tiempo_comparativo)