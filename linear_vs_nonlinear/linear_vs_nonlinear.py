import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
import warnings, sys, io
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


# ============================================================
# SECTION 1 -- PARAMETERS
# ============================================================
m=1.0; K=10.0; Kc=2.0; L=1.0; Lc=1.0; b=0.1
dt=0.005; T_total=15.0
q0=np.array([0.3,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0])
dq0=np.zeros(8)
F_drive=0.5; omega_d=3.5
Fx=np.array([F_drive,0.,0.,0.]); Fy=np.zeros(4)
phi_x=np.zeros(4); phi_y=np.zeros(4)


# ============================================================
# Log: prints to terminal AND collects for text file
# ============================================================
_log_lines = []
def log(msg=""):
    print(msg)
    _log_lines.append(msg)


def save_log(filename="simulation_log.txt"):
    with open(filename, "w") as f:
        f.write("\n".join(_log_lines))
    print(f"Saved: {filename}")


# ============================================================
# SECTION 2 -- PHYSICS
# ============================================================
def spring_pe(dx,dy,L0,stiff):
    r=np.sqrt(dx*dx+dy*dy); return 0.5*stiff*(r-L0)**2


def potential(q):
    x1,y1,x2,y2,x3,y3,x4,y4=q
    V =spring_pe(L+x1,y1,L,K)+spring_pe(x1,L-y1,L,K)
    V+=spring_pe(L-x2,y2,L,K)+spring_pe(x2,L-y2,L,K)
    V+=spring_pe(L+x3,y3,L,K)+spring_pe(x3,L+y3,L,K)
    V+=spring_pe(L-x4,y4,L,K)+spring_pe(x4,L+y4,L,K)
    V+=spring_pe(Lc+x2-x1,y2-y1,Lc,Kc)+spring_pe(Lc+x4-x3,y4-y3,Lc,Kc)
    V+=spring_pe(x3-x1,y3-y1-Lc,Lc,Kc)+spring_pe(x4-x2,y4-y2-Lc,Lc,Kc)
    return V


def _sf(dx,dy,L0,stiff):
    r=np.sqrt(dx*dx+dy*dy)
    if r<1e-14: return 0.,0.
    c=stiff*(r-L0)/r; return -c*dx,-c*dy


def forces_nl(q,t):
    x1,y1,x2,y2,x3,y3,x4,y4=q; F=np.zeros(8)
    def wf(i,dx,dy):
        fx,fy=_sf(dx,dy,L,K); F[2*i]+=fx; F[2*i+1]+=fy
    wf(0,L+x1,y1); wf(0,x1,y1-L)
    wf(1,x2-L,y2); wf(1,x2,y2-L)
    wf(2,L+x3,y3); wf(2,x3,y3+L)
    wf(3,x4-L,y4); wf(3,x4,y4+L)
    def cf(i,j,edx,edy):
        dx=edx+q[2*j]-q[2*i]; dy=edy+q[2*j+1]-q[2*i+1]
        r=np.sqrt(dx*dx+dy*dy)
        if r<1e-14: return
        c=Kc*(r-Lc)/r
        F[2*i]+=c*dx; F[2*i+1]+=c*dy; F[2*j]-=c*dx; F[2*j+1]-=c*dy
    cf(0,1,Lc,0); cf(2,3,Lc,0); cf(0,2,0,-Lc); cf(1,3,0,-Lc)
    F/=m
    for i in range(4):
        F[2*i]+=Fx[i]*np.cos(omega_d*t+phi_x[i])/m
        F[2*i+1]+=Fy[i]*np.cos(omega_d*t+phi_y[i])/m
    return F


K_lin_global=None


def build_stiffness():
    n=8; eps=1e-5; Km=np.zeros((n,n)); qz=np.zeros(n)
    Fx_s,Fy_s=Fx.copy(),Fy.copy(); Fx[:]=0; Fy[:]=0
    def sf(qq): return forces_nl(qq,0.0)*m
    for j in range(n):
        qp=qz.copy(); qp[j]+=eps; qm=qz.copy(); qm[j]-=eps
        Km[:,j]=-(sf(qp)-sf(qm))/(2*eps)
    Fx[:]=Fx_s; Fy[:]=Fy_s; return Km


def forces_lin(q,t):
    F=-(K_lin_global@q)/m
    for i in range(4):
        F[2*i]+=Fx[i]*np.cos(omega_d*t+phi_x[i])/m
        F[2*i+1]+=Fy[i]*np.cos(omega_d*t+phi_y[i])/m
    return F


def rk4_step(q,v,t,dt,force_fn):
    def a(q_,v_,t_): return force_fn(q_,t_)-(b/m)*v_
    k1q=v; k1v=a(q,v,t)
    k2q=v+0.5*dt*k1v; k2v=a(q+0.5*dt*k1q,v+0.5*dt*k1v,t+0.5*dt)
    k3q=v+0.5*dt*k2v; k3v=a(q+0.5*dt*k2q,v+0.5*dt*k2v,t+0.5*dt)
    k4q=v+dt*k3v; k4v=a(q+dt*k3q,v+dt*k3v,t+dt)
    return (q+(dt/6)*(k1q+2*k2q+2*k3q+k4q),v+(dt/6)*(k1v+2*k2v+2*k3v+k4v))


def run_simulation(q_init,v_init,force_fn,label,pot_fn):
    N=int(T_total/dt); Q=np.zeros((N,8)); V=np.zeros((N,8))
    Ek=np.zeros(N); Ep=np.zeros(N); t_arr=np.arange(N)*dt
    q,v=q_init.copy(),v_init.copy()
    log(f"  Running {label} ({N} steps)...")
    for i in range(N):
        Q[i]=q; V[i]=v; Ek[i]=0.5*m*np.dot(v,v); Ep[i]=pot_fn(q)
        q,v=rk4_step(q,v,t_arr[i],dt,force_fn)
    if b==0 and np.max(np.abs(Fx))==0 and np.max(np.abs(Fy))==0:
        E0=Ek[0]+Ep[0]
        drift=abs((Ek[-1]+Ep[-1])-E0)/(abs(E0)+1e-14)
        log(f"    [{label}] Energy drift (undamped): {drift:.3e}")
    return t_arr,Q,V,Ek,Ep


# ============================================================
# SECTION 3 -- NORMAL MODE ANALYSIS  (printed + logged)
# ============================================================
def linear_analysis():
    global K_lin_global
    log("\n"+"="*60)
    log("  NORMAL MODE ANALYSIS")
    log("="*60)
    K_mat=build_stiffness()
    eigvals,eigvecs=eigh(K_mat,m*np.eye(8))
    eigvals=np.maximum(eigvals,0.0); omegas=np.sqrt(eigvals)
    freqs=omegas/(2*np.pi)
    log(f"\n{'Mode':>5}  {'omega [rad/s]':>14}  {'f [Hz]':>10}  {'T [s]':>8}")
    log("-"*50)
    for i,(w,f) in enumerate(zip(omegas,freqs)):
        T=1/f if f>1e-6 else np.inf
        log(f"  {i+1:>3}   {w:>14.5f}   {f:>10.5f}  {T:>8.4f}")
    K_lin_global=K_mat
    return K_mat,omegas,eigvecs


# ============================================================
# SECTION 4 -- PHYSICS VERIFICATION  (printed + logged)
# ============================================================
def physics_checks(omegas,K_mat,eigvecs):
    log("\n"+"="*60)
    log("  PHYSICS VERIFICATION")
    log("="*60)


    asym=np.max(np.abs(K_mat-K_mat.T))
    log(f"\n[1] K symmetry error:          {asym:.2e}  (expect ~0)")


    min_eig=np.min(np.linalg.eigvalsh(K_mat))
    log(f"[2] Min eigenvalue of K:       {min_eig:.4e}  (expect >= 0)")


    off=np.max(np.abs(eigvecs.T@(m*np.eye(8))@eigvecs-np.eye(8)))
    log(f"[3] Eigenvector M-orthogon.:   {off:.2e}  (expect ~1e-16)")


    eps=1e-5
    def V1h(x,y): return spring_pe(L+x,y,L,K)
    def V1v(x,y): return spring_pe(x,L-y,L,K)
    Kxx_h=(V1h(eps,0)+V1h(-eps,0)-2*V1h(0,0))/eps**2
    Kyy_v=(V1v(0,eps)+V1v(0,-eps)-2*V1v(0,0))/eps**2
    log(f"[4] Wall spring d2V/dx2 (horiz): {Kxx_h:.4f}  (expect {K})")
    log(f"    Wall spring d2V/dy2 (vert):  {Kyy_v:.4f}  (expect {K})")


    A_small=1e-4; qt=np.zeros(8); qt[0]=A_small
    Fx_s,Fy_s=Fx.copy(),Fy.copy(); Fx[:]=0; Fy[:]=0
    a_nl=forces_nl(qt,0.0); a_lin=forces_lin(qt,0.0)
    Fx[:]=Fx_s; Fy[:]=Fy_s
    err=np.linalg.norm(a_nl-a_lin)/(np.linalg.norm(a_nl)+1e-14)
    log(f"[5] NL vs Lin at A={A_small}: rel err = {err:.2e}  (expect < 1e-3)")


    qt2=np.zeros(8); qt2[0]=0.2
    Fx_s,Fy_s=Fx.copy(),Fy.copy(); Fx[:]=0; Fy[:]=0
    F2=forces_nl(qt2,0.0)*m; Fx[:]=Fx_s; Fy[:]=Fy_s
    ok1="OK" if F2[0]<0 else "WRONG"
    log(f"[6] m1 x=+0.2 -> Fx={F2[0]:.4f} ({ok1}, expect <0 restores left)")


    qt3=np.zeros(8); qt3[1]=0.2
    Fx_s,Fy_s=Fx.copy(),Fy.copy(); Fx[:]=0; Fy[:]=0
    F3=forces_nl(qt3,0.0)*m; Fx[:]=Fx_s; Fy[:]=Fy_s
    ok2="OK" if F3[1]<0 else "WRONG"
    log(f"    m1 y=+0.2 -> Fy={F3[1]:.4f} ({ok2}, expect <0 restores down)")


    qt4=np.zeros(8); qt4[0]=-0.1; qt4[2]=0.1
    Fx_s,Fy_s=Fx.copy(),Fy.copy(); Fx[:]=0; Fy[:]=0
    F4=forces_nl(qt4,0.0)*m; Fx[:]=Fx_s; Fy[:]=Fy_s
    ok3="OK" if F4[0]>0 and F4[2]<0 else "WRONG"
    log(f"[7] m1-m2 stretched: Fx1={F4[0]:.4f}(>0), Fx2={F4[2]:.4f}(<0)  ({ok3})")


    log(f"\n[8] 1D chain benchmark (K={K}, Kc={Kc}, m={m}):")
    log(f"    omega_in_phase     = {np.sqrt(K/m):.4f} rad/s")
    log(f"    omega_out_of_phase = {np.sqrt((K+2*Kc)/m):.4f} rad/s")
    log(f"    Actual mode range: {omegas[0]:.4f} to {omegas[-1]:.4f} rad/s")
    log(f"    Note: 2D system modes span this range due to square symmetry")


# ============================================================
# MAIN: header + run
# ============================================================
log("="*60)
log("  2x2 COUPLED MASS-SPRING SYSTEM")
log(f"  m={m}  K={K}  Kc={Kc}  L={L}  Lc={Lc}  b={b}")
log(f"  dt={dt}  T={T_total}s")
log(f"  q0={q0}")
log(f"  Fx={Fx}  Fy={Fy}  omega_d={omega_d} rad/s")
log("="*60)


K_mat,omegas,eigvecs = linear_analysis()
physics_checks(omegas,K_mat,eigvecs)


def pe_lin(q): return 0.5*q@K_lin_global@q


log("\n"+"="*60); log("  RK4 INTEGRATION"); log("="*60)
t,Q_nl,V_nl,Ek_nl,Ep_nl   = run_simulation(q0,dq0,forces_nl, "Nonlinear",potential)
t,Q_lin,V_lin,Ek_lin,Ep_lin = run_simulation(q0,dq0,forces_lin,"Linear",   pe_lin)


# ============================================================
# PLOTS
# ============================================================
log("\n"+"="*60); log("  GENERATING PLOTS"); log("="*60)
CNL="#1f77b4"; CLIN="#d62728"
NOISE_THRESH=1e-6


# --- 01: Displacement ---
labs=["x1","y1","x2","y2","x3","y3","x4","y4"]
fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
fig.suptitle("Displacement vs Time  —  Nonlinear (solid) vs Linear (dashed)",fontsize=11)
for i in range(8):
    ax=axes[i//2,i%2]
    ax.plot(t,Q_nl[:,i],lw=0.9,color=CNL,label="Nonlinear")
    ax.plot(t,Q_lin[:,i],lw=0.9,color=CLIN,ls="--",label="Linear")
    ax.set_ylabel(labs[i]+" [m]"); ax.legend(fontsize=7); ax.grid(lw=0.4,alpha=0.5)
axes[-1,0].set_xlabel("t [s]"); axes[-1,1].set_xlabel("t [s]")
plt.tight_layout(); plt.savefig("01_displacement_vs_time.png"); plt.close()
log("Saved: 01_displacement_vs_time.png")


# --- 02: Phase portraits (separate NL / Lin, noise flagged) ---
fig,axes=plt.subplots(4,4,figsize=(17,13))
fig.suptitle("Phase Portraits  —  Cols 1-2: Nonlinear  |  Cols 3-4: Linear\n"
             "Grey = numerically zero (amplitude < 1e-6 m)",fontsize=10)
col_titles=["NL  x-phase (x,vx)","NL  y-phase (y,vy)",
            "Lin  x-phase (x,vx)","Lin  y-phase (y,vy)"]
for j,ct in enumerate(col_titles):
    axes[0,j].set_title(ct,fontsize=9,fontweight="bold",color=CNL if j<2 else CLIN)
for i in range(4):
    datasets=[
        (Q_nl[:,2*i],  V_nl[:,2*i],  CNL, f"x{i+1}",f"vx{i+1}"),
        (Q_nl[:,2*i+1],V_nl[:,2*i+1],CNL, f"y{i+1}",f"vy{i+1}"),
        (Q_lin[:,2*i], V_lin[:,2*i], CLIN,f"x{i+1}",f"vx{i+1}"),
        (Q_lin[:,2*i+1],V_lin[:,2*i+1],CLIN,f"y{i+1}",f"vy{i+1}"),
    ]
    for j,(pos,vel,col,xl,yl) in enumerate(datasets):
        ax=axes[i,j]; amp=np.max(np.abs(pos)); is_noise=amp<NOISE_THRESH
        if is_noise:
            ax.plot(pos,vel,lw=0.4,color="#aaaaaa")
            ax.text(0.5,0.5,f"numerical noise\namp ~ {amp:.1e} m",
                    transform=ax.transAxes,ha="center",va="center",fontsize=8,
                    color="#888888",bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white",edgecolor="#cccccc",alpha=0.8))
            ax.set_facecolor("#f9f9f9")
        else:
            ax.plot(pos,vel,lw=0.5,color=col)
        ax.set_xlabel(xl+" [m]",fontsize=8); ax.set_ylabel(yl+" [m/s]",fontsize=8)
        ax.text(0.04,0.93,f"m{i+1}",transform=ax.transAxes,fontsize=9,fontweight="bold")
        ax.grid(lw=0.4,alpha=0.4)
plt.tight_layout(); plt.savefig("02_phase_portraits15.png"); plt.close()
log("Saved: 02_phase_portraits.png")


# --- 03: Energy ---
fig,axes=plt.subplots(1,2,figsize=(12,4))
fig.suptitle("Energy vs Time",fontsize=11)
for ax,(Ek,Ep,lbl) in zip(axes,[(Ek_nl,Ep_nl,"Nonlinear"),(Ek_lin,Ep_lin,"Linear")]):
    Et=Ek+Ep
    ax.plot(t,Ek,lw=0.8,color=CNL,label="KE")
    ax.plot(t,Ep,lw=0.8,color=CLIN,label="PE")
    ax.plot(t,Et,lw=1.2,color="#2ca02c",ls="--",label="Total")
    ax.set_title(lbl); ax.set_xlabel("t [s]"); ax.set_ylabel("E [J]")
    ax.legend(fontsize=8); ax.grid(lw=0.4,alpha=0.5)
    drift=(Et.max()-Et.min())/(abs(Et[0])+1e-14)*100
    ax.text(0.02,0.05,f"E drift={drift:.3f}%",transform=ax.transAxes,fontsize=8)
    log(f"  [{lbl}] Energy drift = {drift:.4f}%")
plt.tight_layout(); plt.savefig("03_energy_vs_time.png"); plt.close()
log("Saved: 03_energy_vs_time.png")


# --- 04: FFT ---
N_fft  = len(t)
window = np.hanning(N_fft)
fhz    = np.fft.rfftfreq(N_fft, d=dt)


for Q, lbl, col, fname in [
        (Q_nl,  "Nonlinear", CNL,  "04a_fft_nonlinear.png"),
        (Q_lin, "Linear",    CLIN, "04b_fft_linear.png"),
]:
    # 4 masses × 2 DOFs (x/y) → 2 rows (x top, y bottom) × 4 cols (masses)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"FFT Spectrum — {lbl} (Hann window)  |  red dashed = normal mode freqs",
        fontsize=11
    )
    axes[0, 0].set_ylabel("x-DOF  |  norm. |FFT|", fontsize=9)
    axes[1, 0].set_ylabel("y-DOF  |  norm. |FFT|", fontsize=9)


    for i in range(4):                              # mass
        for row, (dof_idx, dof_lbl) in enumerate([(2*i, "x"), (2*i+1, "y")]):
            ax   = axes[row, i]
            sig  = Q[:, dof_idx] - np.mean(Q[:, dof_idx])
            spec = np.abs(np.fft.rfft(sig * window))
            spec /= spec.max() + 1e-14


            ax.plot(fhz, spec, lw=0.8, color=col)


            for k, w in enumerate(omegas):
                ax.axvline(w / (2*np.pi), color="r", lw=0.7, ls="--", alpha=0.7,
                           label=f"ω{k+1}" if (i == 0 and row == 0) else None)


            ax.set_title(f"m{i+1}  {dof_lbl}-DOF", fontsize=9)
            ax.set_xlim(0, omegas[-1] / (2*np.pi) * 1.4)
            ax.set_ylim(0, 1.05)
            ax.grid(lw=0.4, alpha=0.5)
            if row == 1:
                ax.set_xlabel("f [Hz]", fontsize=8)


    axes[0, 0].legend(ncol=4, fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    log(f"Saved: {fname}")
# --- 05: Modal energy ---
V_inv=np.linalg.inv(eigvecs); modal=Q_nl@V_inv.T
fig,axes=plt.subplots(2,4,figsize=(16,7))
fig.suptitle("Modal Energy Transfer — NL trajectory on linear eigenmodes",fontsize=11)
for i in range(8):
    ax=axes[i//4,i%4]; mode_E=modal[:,i]**2
    ax.plot(t,mode_E,lw=0.7,color=f"C{i}")
    ax.set_title(f"Mode {i+1}  ω={omegas[i]:.3f} rad/s",fontsize=9)
    ax.set_xlabel("t [s]"); ax.set_ylabel("amplitude²"); ax.grid(lw=0.4,alpha=0.5)
    ax.axhline(np.mean(mode_E),color="k",lw=0.8,ls="--",alpha=0.6,
               label=f"mean={np.mean(mode_E):.3f}"); ax.legend(fontsize=7)
    log(f"  Mode {i+1} mean modal energy = {np.mean(mode_E):.6f}")
plt.tight_layout(); plt.savefig("05_modal_energy.png"); plt.close()
log("Saved: 05_modal_energy.png")


# --- 06: Frequency response ---
omega_range=np.linspace(0.05,omegas[-1]*1.6,800)
F_vec=np.zeros(8)
for i in range(4): F_vec[2*i]=Fx[i]; F_vec[2*i+1]=Fy[i]
if np.max(np.abs(F_vec))<1e-14: F_vec[0]=1.0
A_arr=[]
for w in omega_range:
    Z=K_lin_global-w**2*m*np.eye(8)+1j*b*w*np.eye(8)
    try: A_arr.append(np.max(np.abs(np.linalg.solve(Z,F_vec))))
    except: A_arr.append(np.nan)
fig,ax=plt.subplots(figsize=(10,5))
ax.semilogy(omega_range/(2*np.pi),np.array(A_arr),lw=1.2,color=CNL)
for k,w in enumerate(omegas):
    ax.axvline(w/(2*np.pi),color="r",lw=0.7,ls="--",alpha=0.7,
               label=f"w{k+1}" if k<4 else None)
ax.set_xlabel("Drive frequency [Hz]"); ax.set_ylabel("Max amplitude [m]")
ax.set_title("Frequency Response (linear, impedance method)")
ax.legend(ncol=4,fontsize=8); ax.grid(lw=0.4,alpha=0.5,which="both")
plt.tight_layout(); plt.savefig("06_frequency_response.png"); plt.close()
log("Saved: 06_frequency_response.png")


# --- 07: Frequency shift ---
log("  Computing nonlinear frequency shift...")
T_sw = 80.0  # longer window for better freq resolution
N_s = int(T_sw / dt)


# More data points: fine spacing at low amplitude, coarser at high
amps_low  = np.linspace(0.005, 0.1,  20)   # fine: capture onset of nonlinearity
amps_mid  = np.linspace(0.1,   0.5,  20)   # medium
amps_high = np.linspace(0.5,   1.2,  15)   # high amplitude, coarser
amps = np.unique(np.concatenate([amps_low, amps_mid, amps_high]))


peak_f = []
Fx_s, Fy_s = Fx.copy(), Fy.copy(); Fx[:] = 0; Fy[:] = 0


for A in amps:
    qa = np.zeros(8); qa[0] = A
    qq, vv = qa.copy(), np.zeros(8)
    Qa = np.zeros(N_s)
    for i in range(N_s):
        Qa[i] = qq[0]
        qq, vv = rk4_step(qq, vv, i * dt, dt, forces_nl)


    win = np.hanning(N_s)
    ff  = np.fft.rfftfreq(N_s, d=dt)
    sp  = np.abs(np.fft.rfft((Qa - Qa.mean()) * win)); sp[0] = 0


    # Parabolic interpolation around the peak for sub-bin freq resolution
    idx = np.argmax(sp)
    if 1 <= idx <= len(sp) - 2:
        alpha, beta, gamma = sp[idx-1], sp[idx], sp[idx+1]
        delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-30)
        pf = ff[idx] + delta * (ff[1] - ff[0])
    else:
        pf = ff[idx]


    peak_f.append(pf)
    log(f"    A={A:.4f}  ->  f_peak={pf:.6f} Hz")


Fx[:] = Fx_s; Fy[:] = Fy_s
peak_f = np.array(peak_f)
f_lin  = omegas[0] / (2 * np.pi)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Nonlinear Frequency Shift vs Amplitude", fontsize=12)


# Left: absolute frequency
ax = axes[0]
ax.plot(amps, peak_f, "o-", lw=1.2, ms=4, color=CNL, label="NL peak freq")
ax.axhline(f_lin, color="r", ls="--", lw=1.0,
           label=f"Linear f₁ = {f_lin:.5f} Hz")
ax.set_xlabel("Initial amplitude x₁(0) [m]")
ax.set_ylabel("Peak frequency [Hz]")
ax.legend(); ax.grid(lw=0.4, alpha=0.5)


# Right: relative shift in %
ax2 = axes[1]
shift_pct = (peak_f - f_lin) / f_lin * 100
ax2.plot(amps, shift_pct, "s-", lw=1.2, ms=4, color="#ff7f0e",
         label="(f_NL - f_lin) / f_lin × 100")
ax2.axhline(0, color="r", ls="--", lw=0.9)
ax2.set_xlabel("Initial amplitude x₁(0) [m]")
ax2.set_ylabel("Frequency shift [%]")
ax2.legend(); ax2.grid(lw=0.4, alpha=0.5)


plt.tight_layout()
plt.savefig("07_freq_shift.png", dpi=130)
plt.close()
log("Saved: 07_freq_shift.png")


# --- 08: Normal modes GIF ---
# def make_normal_modes_gif(K, Kc, L, Lc, m):
#     log("\nGenerating 08_normal_modes.gif...")
#     raw_modes=np.array([
#         [ 1,0, 1,0, 1,0, 1,0],  # A: all +x
#         [ 0,1, 0,1, 0,1, 0,1],  # B: all +y
#         [ 1,0, 1,0,-1,0,-1,0],  # E: x shear top/bot
#         [ 0,1, 0,-1,0,1, 0,-1], # F: y shear left/right
#         [ 1,0,-1,0, 1,0,-1,0],  # C: x antisym L/R
#         [ 0,1, 0,1, 0,-1,0,-1], # D: y antisym T/B
#         [ 1,0,-1,0,-1,0, 1,0],  # G: x breathing
#         [ 0,1, 0,-1,0,-1,0,1],  # H: y breathing
#     ],dtype=float)
#     for i in range(8): raw_modes[i]/=np.linalg.norm(raw_modes[i])
#     w1=np.sqrt(K/m); w2=np.sqrt((K+2*Kc)/m)
#     omegas_phys=np.array([w1,w1,w1,w1,w2,w2,w2,w2])
#     mode_names=["A: all +x (translation)","B: all +y (translation)",
#                 "E: x shear (top vs bottom)","F: y shear (left vs right)",
#                 "C: x antisym (left vs right)","D: y antisym (top vs bottom)",
#                 "G: x breathing (diagonal)","H: y breathing (diagonal)"]
#     d2=Lc/2
#     EQ=np.array([[-d2,d2],[d2,d2],[-d2,-d2],[d2,-d2]])
#     ANCHOR_H=np.array([[-d2-L,d2],[d2+L,d2],[-d2-L,-d2],[d2+L,-d2]])
#     ANCHOR_V=np.array([[-d2,d2+L],[d2,d2+L],[-d2,-d2-L],[d2,-d2-L]])
#     pad=L+0.6
#     fig,ax=plt.subplots(figsize=(7,7))
#     ax.set_aspect("equal")
#     ax.set_xlim(EQ[:,0].min()-pad,EQ[:,0].max()+pad)
#     ax.set_ylim(EQ[:,1].min()-pad,EQ[:,1].max()+pad)
#     ax.axis("off"); title_txt=ax.set_title("",fontsize=10)
#     def draw_wall(cx,cy,ori):
#         W=0.2;T=0.05;H=0.1
#         if ori in ['left','right']:
#             ax.add_patch(plt.Rectangle((cx-T/2,cy-W),T,2*W,color='black',zorder=5))
#             for yo in np.linspace(-W,W,5):
#                 hx=cx-T/2 if ori=='left' else cx+T/2; hd=-1 if ori=='left' else 1
#                 ax.plot([hx,hx+hd*H],[cy+yo,cy+yo-H],'k-',lw=1)
#         else:
#             ax.add_patch(plt.Rectangle((cx-W,cy-T/2),2*W,T,color='black',zorder=5))
#             for xo in np.linspace(-W,W,5):
#                 hy=cy+T/2 if ori=='up' else cy-T/2; hd=1 if ori=='up' else -1
#                 ax.plot([cx+xo,cx+xo-H],[hy,hy+hd*H],'k-',lw=1)
#     for i in range(4):
#         draw_wall(ANCHOR_H[i,0],ANCHOR_H[i,1],['left','right','left','right'][i])
#         draw_wall(ANCHOR_V[i,0],ANCHOR_V[i,1],['up','up','down','down'][i])
#     coupling_pairs=[(0,1),(2,3),(0,2),(1,3)]
#     mass_dots=ax.scatter([],[],s=160,color='black',zorder=10)
#     coup_lines=[ax.plot([],[],'b-',lw=1.8,zorder=3)[0] for _ in range(4)]
#     wall_lines=[ax.plot([],[],'g-',lw=1.3,zorder=3)[0] for _ in range(8)]
#     quiv=ax.quiver(EQ[:,0],EQ[:,1],np.zeros(4),np.zeros(4),
#                    color='red',scale=1,scale_units='xy',angles='xy',width=0.012,zorder=8)
#     n_modes=8; nf=120; amp=0.28
#     tc=np.linspace(0,2*np.pi,nf,endpoint=False)
#     def update(frame):
#         mi=frame//nf; fi=frame%nf
#         disp=amp*raw_modes[mi]*np.cos(tc[fi]); pos=EQ.copy()
#         for i in range(4): pos[i,0]+=disp[2*i]; pos[i,1]+=disp[2*i+1]
#         mass_dots.set_offsets(pos)
#         for k,(i,j) in enumerate(coupling_pairs):
#             coup_lines[k].set_data([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]])
#         for i in range(4):
#             wall_lines[2*i  ].set_data([ANCHOR_H[i,0],pos[i,0]],[ANCHOR_H[i,1],pos[i,1]])
#             wall_lines[2*i+1].set_data([ANCHOR_V[i,0],pos[i,0]],[ANCHOR_V[i,1],pos[i,1]])
#         U=np.array([raw_modes[mi][2*i]   for i in range(4)])*amp*0.7
#         V=np.array([raw_modes[mi][2*i+1] for i in range(4)])*amp*0.7
#         quiv.set_UVC(U,V)
#         w=omegas_phys[mi]
#         title_txt.set_text(f"Mode {mode_names[mi]}\nω={w:.4f} rad/s   f={w/(2*np.pi):.4f} Hz")
#         return [mass_dots,*coup_lines,*wall_lines,title_txt,quiv]
#     anim=animation.FuncAnimation(fig,update,frames=n_modes*nf,interval=30,blit=True)
#     anim.save("08_normal_modes.gif",writer=animation.PillowWriter(fps=30))
#     plt.close()
#     log("Saved: 08_normal_modes.gif")


# make_normal_modes_gif(K,Kc,L,Lc,m)


# ============================================================
# SAVE LOG FILE
# ============================================================
log("\n"+"="*60)
log("  ALL DONE")
log("  Output files:")
for f in ["01_displacement_vs_time.png","02_phase_portraits.png",
          "03_energy_vs_time.png","04_fft_spectrum.png","05_modal_energy.png",
          "06_frequency_response.png","07_freq_shift.png","08_normal_modes.gif",
          "simulation_log.txt"]:
    log(f"    {f}")
log("="*60)
save_log("simulation_log.txt")


