#!/usr/bin/env python
"""Phase 2 Visualization for D2NN Loss Strategy Sweep.

Figure 1: Input field
Figure 2: D2NN output (vacuum, turb, completed configs)
Figure 3: Detector plane (PIB visualization)
Figure 4: Phase masks
Figure 5: Training curves (loss, PIB, CO)
Figure 6: Metric comparison bar chart

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_loss_strategy_report.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap

W=1.55e-6; N=1024; WIN=0.002048; APT=0.002; DX=WIN/N; FOCUS_F=4.5e-3
ARCH=dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)
ALL_CONFIGS=["pib_only","strehl_only","intensity_overlap","co_pib_hybrid"]

DATA_DIR=Path(__file__).resolve().parent.parent/"data"/"kim2026"/"1km_cn2_5e-14_tel15cm_n1024_br75"
SWEEP_DIR=Path(__file__).resolve().parent.parent/"autoresearch"/"runs"/"d2nn_loss_strategy"
OUT=SWEEP_DIR

def load_model(name):
    ckpt=SWEEP_DIR/name/"checkpoint.pt"
    if not ckpt.exists(): return None
    m=BeamCleanupD2NN(n=N,wavelength_m=W,window_m=WIN,**ARCH)
    m.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True)["model_state_dict"])
    m.eval(); return m

def prepare(f): return center_crop_field(apply_receiver_aperture(f,receiver_window_m=WIN,aperture_diameter_m=APT),crop_n=N)

def focus(field):
    with torch.no_grad():
        f,dx=lens_2f_forward(field.to(torch.complex64),dx_in_m=DX,wavelength_m=W,f_m=FOCUS_F,na=None,apply_scaling=False)
    return f,dx

def ext(n,dx,u): h=n*dx/2/u; return [-h,h,-h,h]

def compute_pib(field, dx, radius_um=50):
    irr=np.abs(field)**2
    n=field.shape[-1]; c=n//2
    yy,xx=np.mgrid[-c:n-c,-c:n-c]
    r=np.sqrt((xx*dx)**2+(yy*dx)**2)
    mask=r<=(radius_um*1e-6)
    return irr[mask].sum()/max(irr.sum(),1e-30)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds=CachedFieldDataset(cache_dir=str(DATA_DIR/"cache"),manifest_path=str(DATA_DIR/"split_manifest.json"),split="test")
    s=ds[0]
    u_turb=prepare(s["u_turb"].unsqueeze(0).to(device))
    u_vac=prepare(s["u_vacuum"].unsqueeze(0).to(device))

    # Load completed
    configs_done=[]; outputs={}
    for name in ALL_CONFIGS:
        model=load_model(name)
        if model:
            model=model.to(device)
            with torch.no_grad(): outputs[name]=model(u_turb)
            co=complex_overlap(outputs[name],u_vac).item()
            print(f"  {name:>20}: CO={co:.4f}")
            configs_done.append(name)
        else:
            print(f"  {name:>20}: not completed")

    # No correction baseline
    d0=BeamCleanupD2NN(n=N,wavelength_m=W,window_m=WIN,**ARCH).to(device); d0.eval()
    with torch.no_grad(): u_vac_d=d0(u_vac); u_turb_d=d0(u_turb)

    # Focus all
    u_vac_det,dx_det=focus(u_vac_d); u_turb_det,_=focus(u_turb_d)
    det_out={n:focus(o)[0] for n,o in outputs.items()}

    # numpy
    vac_np=u_vac[0].cpu().numpy(); turb_np=u_turb[0].cpu().numpy()
    vac_d_np=u_vac_d[0].cpu().numpy(); turb_d_np=u_turb_d[0].cpu().numpy()
    out_np={n:o[0].cpu().numpy() for n,o in outputs.items()}
    vac_det=u_vac_det[0].cpu().numpy(); turb_det=u_turb_det[0].cpu().numpy()
    det_np={n:o[0].cpu().numpy() for n,o in det_out.items()}
    e_mm=ext(N,DX,1e-3); mid=N//2; x_mm=np.linspace(e_mm[0],e_mm[1],N)

    # ═══ FIGURE 1: Input ═══
    print("\nFig 1...")
    fig1,ax1=plt.subplots(1,4,figsize=(24,6))
    fig1.suptitle("Figure 1: Input Field (Cn²=5e-14, D/r₀=5.0)",fontsize=16,fontweight="bold")
    imax=(np.abs(vac_np)**2).max()
    ax1[0].imshow(np.abs(vac_np)**2,extent=e_mm,origin="lower",cmap="inferno",vmin=0,vmax=imax)
    ax1[0].set_title("Vacuum irradiance")
    ax1[1].imshow(np.angle(vac_np),extent=e_mm,origin="lower",cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
    ax1[1].set_title("Vacuum phase")
    ax1[2].imshow(np.abs(turb_np)**2,extent=e_mm,origin="lower",cmap="inferno",vmin=0,vmax=imax)
    ax1[2].set_title("Turbulent irradiance")
    ax1[3].imshow(np.angle(turb_np),extent=e_mm,origin="lower",cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
    ax1[3].set_title("Turbulent phase")
    plt.tight_layout(rect=[0,0,1,0.93])
    fig1.savefig(OUT/"phase2_fig1_input.png",dpi=150,bbox_inches="tight"); plt.close(fig1)
    print("  Saved fig1")

    # ═══ FIGURE 2: D2NN Output (detector plane, after last layer + 10mm propagation) ═══
    print("Fig 2...")
    cols=[("Vacuum",vac_d_np),("Turb\n(no D2NN)",turb_d_np)]
    for c in configs_done: cols.append((c,out_np[c]))
    ncols=len(cols)
    # Use μm for detector plane to match bucket scale
    e_um_det=ext(N,DX,1e-6)  # extent in μm
    x_um=np.linspace(e_um_det[0],e_um_det[1],N)
    fig2,ax2=plt.subplots(3,ncols,figsize=(6*ncols,18))
    plane_label = "D2NN output plane (after 5 layers + 10 mm prop.) — BEFORE focus lens"
    fig2.suptitle(f"Figure 2: D2NN Output — Loss Strategy Comparison\n{plane_label}  ⚠ NOT the detector focal plane (see Fig 3)\nWindow: {WIN*1e3:.3f} mm × {WIN*1e3:.3f} mm, dx = {DX*1e6:.1f} μm, {N}×{N} pixels",
                  fontsize=14,fontweight="bold")
    for col,(label,field) in enumerate(cols):
        irr=np.abs(field)**2
        log_irr=np.log10(irr+1e-30)
        ax2[0,col].imshow(log_irr,extent=e_um_det,origin="lower",cmap="inferno",vmin=log_irr.max()-6,vmax=log_irr.max())
        ax2[0,col].set_title(label,fontsize=11,fontweight="bold")
        if col==0:
            ax2[0,col].set_xlabel("μm"); ax2[0,col].set_ylabel("Irradiance (log₁₀)\nμm",fontsize=11,fontweight="bold")
        # 50μm bucket circle
        theta=np.linspace(0,2*np.pi,200)
        ax2[0,col].plot(50*np.cos(theta),50*np.sin(theta),'c--',lw=1.0,alpha=0.8)
        # Scale bar 200μm
        if col==0:
            ax2[0,col].plot([-e_um_det[1]*0.85,-e_um_det[1]*0.85+200],[-e_um_det[1]*0.85,-e_um_det[1]*0.85],'w-',lw=3)
            ax2[0,col].text(-e_um_det[1]*0.85+100,-e_um_det[1]*0.78,"200 μm",color='w',fontsize=9,ha='center')

        ax2[1,col].imshow(np.angle(field),extent=e_um_det,origin="lower",cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
        if col==0: ax2[1,col].set_ylabel("Phase [rad]\nμm",fontsize=11,fontweight="bold")

        ax2[2,col].plot(x_um,np.abs(vac_d_np[mid,:])**2,'b--',lw=1,alpha=0.5,label='Vacuum')
        ax2[2,col].plot(x_um,np.abs(field[mid,:])**2,'r-',lw=1.5,label=label)
        ax2[2,col].set_xlabel("μm")
        ax2[2,col].axvline(-50,color='cyan',ls='--',lw=0.8,alpha=0.5)
        ax2[2,col].axvline(50,color='cyan',ls='--',lw=0.8,alpha=0.5)
        ax2[2,col].legend(fontsize=8); ax2[2,col].grid(True,alpha=0.3)
        if col==0: ax2[2,col].set_ylabel("1D profile\nIrradiance",fontsize=11,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.92])
    fig2.savefig(OUT/"phase2_fig2_output.png",dpi=150,bbox_inches="tight"); plt.close(fig2)
    print("  Saved fig2")

    # ═══ FIGURE 3: Detector + PIB ═══
    print("Fig 3...")
    det_cols=[("Vacuum",vac_det),("Turb\n(no D2NN)",turb_det)]
    for c in configs_done: det_cols.append((c,det_np[c]))
    ncols=len(det_cols)
    fig3,ax3=plt.subplots(3,ncols,figsize=(6*ncols,18))
    fig3.suptitle("Figure 3: Detector Plane — PIB Analysis",fontsize=16,fontweight="bold")
    Z=64; cd=N//2; e_um=ext(2*Z,float(dx_det),1e-6)
    imax_d=max((np.abs(f[cd-Z:cd+Z,cd-Z:cd+Z])**2).max() for _,f in det_cols)
    yy,xx=np.mgrid[-cd:N-cd,-cd:N-cd]; rsq=(xx*float(dx_det))**2+(yy*float(dx_det))**2
    radii=np.linspace(1,100,100)*1e-6

    for col,(label,field) in enumerate(det_cols):
        crop=field[cd-Z:cd+Z,cd-Z:cd+Z]
        ax3[0,col].imshow(np.abs(crop)**2,extent=e_um,origin="lower",cmap="inferno",vmin=0,vmax=imax_d)
        pib=compute_pib(field,float(dx_det),50)
        ax3[0,col].set_title(f"{label}\nPIB@50um={pib:.4f}",fontsize=11,fontweight="bold")
        # 50um circle
        theta=np.linspace(0,2*np.pi,100)
        ax3[0,col].plot(50*np.cos(theta),50*np.sin(theta),'w--',lw=1,alpha=0.7)

        x_um=np.linspace(e_um[0],e_um[1],2*Z)
        ax3[1,col].plot(x_um,np.abs(vac_det[cd,cd-Z:cd+Z])**2,'b--',lw=1,alpha=0.5,label='Vac')
        ax3[1,col].plot(x_um,np.abs(field[cd,cd-Z:cd+Z])**2,'r-',lw=1.5,label=label)
        ax3[1,col].legend(fontsize=8); ax3[1,col].grid(True,alpha=0.3); ax3[1,col].set_xlabel("um")

        irr=np.abs(field)**2; irr_v=np.abs(vac_det)**2
        ee_v=[irr_v[rsq<=r**2].sum()/max(irr_v.sum(),1e-30) for r in radii]
        ee_f=[irr[rsq<=r**2].sum()/max(irr.sum(),1e-30) for r in radii]
        ax3[2,col].plot(radii*1e6,ee_v,'b--',lw=1,alpha=0.5,label='Vac')
        ax3[2,col].plot(radii*1e6,ee_f,'r-',lw=1.5,label=label)
        ax3[2,col].axvline(50,color='gray',ls=':',lw=1)
        ax3[2,col].legend(fontsize=8); ax3[2,col].grid(True,alpha=0.3); ax3[2,col].set_xlabel("Radius [um]")
    for r,lbl in enumerate(["Irradiance\n(+50um circle)","1D cross-section","Encircled energy"]):
        ax3[r,0].set_ylabel(lbl,fontsize=11,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    fig3.savefig(OUT/"phase2_fig3_detector.png",dpi=150,bbox_inches="tight"); plt.close(fig3)
    print("  Saved fig3")

    # ═══ FIGURE 4: Phase masks ═══
    print("Fig 4...")
    nc=len(configs_done)
    if nc > 0:
        fig4,ax4=plt.subplots(6,nc,figsize=(7*nc,36))
        if nc == 1: ax4 = ax4.reshape(-1,1)
        fig4.suptitle("Figure 4: Learned Phase Masks — Loss Strategy",fontsize=16,fontweight="bold")
        for col,name in enumerate(configs_done):
            model=load_model(name)
            if not model: continue
            for li in range(5):
                phase=torch.remainder(model.layers[li].phase,2*math.pi).detach().cpu().numpy()
                ax4[li,col].imshow(phase,cmap="twilight_shifted",vmin=0,vmax=2*math.pi)
                ax4[li,col].set_title(f"{name}\nLayer {li}" if li==0 else f"Layer {li}",fontsize=10)
                ax4[li,col].axis("off")
            spec=np.abs(np.fft.fftshift(np.fft.fft2(torch.remainder(model.layers[0].phase,2*math.pi).detach().cpu().numpy())))**2
            ax4[5,col].imshow(np.log10(spec+1e-10),cmap="viridis"); ax4[5,col].set_title("FFT L0"); ax4[5,col].axis("off")
        plt.tight_layout(rect=[0,0,1,0.97])
        fig4.savefig(OUT/"phase2_fig4_masks.png",dpi=150,bbox_inches="tight"); plt.close(fig4)
        print("  Saved fig4")

    # ═══ FIGURE 5: Training curves ═══
    print("Fig 5...")
    fig5,ax5=plt.subplots(3,1,figsize=(16,18))
    fig5.suptitle("Figure 5: Training Curves — Loss Strategy",fontsize=16,fontweight="bold")
    colors={'pib_only':'#e74c3c','strehl_only':'#3498db','intensity_overlap':'#2ecc71','co_pib_hybrid':'#9b59b6'}
    for name in configs_done:
        rp=SWEEP_DIR/name/"results.json"
        if not rp.exists(): continue
        r=json.load(open(rp)); h=r.get("history",{})
        if not h.get("epoch"): continue
        c=colors.get(name,'gray')
        ax5[0].plot(h["epoch"],h["loss"],color=c,lw=2,marker='o',ms=3,label=name)
        ax5[1].plot(h["epoch"],h["val_co"],color=c,lw=2,marker='o',ms=3,label=name)
        if "val_pib" in h:
            ax5[2].plot(h["epoch"],h["val_pib"],color=c,lw=2,marker='o',ms=3,label=name)
    ax5[1].axhline(0.3044,color='k',ls='--',lw=1.5,label='no_correction CO')
    for a,t,yl in zip(ax5,["Training Loss","Validation CO","Validation PIB@50um"],["Loss","CO","PIB"]):
        a.set_xlabel("Epoch"); a.set_title(t,fontsize=13,fontweight="bold")
        a.legend(fontsize=10); a.grid(True,alpha=0.3); a.set_ylabel(yl)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig5.savefig(OUT/"phase2_fig5_training.png",dpi=150,bbox_inches="tight"); plt.close(fig5)
    print("  Saved fig5")

    # ═══ FIGURE 6: Metric comparison ═══
    print("Fig 6...")
    fig6,ax6=plt.subplots(1,3,figsize=(24,8))
    fig6.suptitle("Figure 6: Loss Strategy — Metric Comparison",fontsize=16,fontweight="bold")
    all_r=[]
    for name in configs_done:
        rp=SWEEP_DIR/name/"results.json"
        if rp.exists(): all_r.append(json.load(open(rp)))
    if all_r:
        names=["no_d2nn"]+[r["name"] for r in all_r]
        # PIB
        pibs=[all_r[0]["baseline_pib_50um"]]+[r["pib_50um"] for r in all_r]
        bar_colors=['gray']+[colors.get(r["name"],'gray') for r in all_r]
        ax6[0].bar(range(len(names)),pibs,color=bar_colors,alpha=0.7)
        ax6[0].set_xticks(range(len(names))); ax6[0].set_xticklabels(names,rotation=30,ha="right")
        ax6[0].set_ylabel("PIB@50um"); ax6[0].set_title("PIB@50um"); ax6[0].grid(True,alpha=0.3,axis='y')
        for i,v in enumerate(pibs): ax6[0].text(i,v+0.01,f"{v:.4f}",ha="center",fontsize=10,fontweight="bold")
        # CO
        cos=[all_r[0]["baseline_co"]]+[r["complex_overlap"] for r in all_r]
        ax6[1].bar(range(len(names)),cos,color=bar_colors,alpha=0.7)
        ax6[1].set_xticks(range(len(names))); ax6[1].set_xticklabels(names,rotation=30,ha="right")
        ax6[1].set_ylabel("CO"); ax6[1].set_title("Complex Overlap"); ax6[1].grid(True,alpha=0.3,axis='y')
        for i,v in enumerate(cos): ax6[1].text(i,v+0.005,f"{v:.4f}",ha="center",fontsize=10)
        # WF RMS
        wfs=[all_r[0].get("wf_rms_baseline_nm",0)]+[r.get("wf_rms_nm",0) for r in all_r]
        ax6[2].bar(range(len(names)),wfs,color=bar_colors,alpha=0.7)
        ax6[2].set_xticks(range(len(names))); ax6[2].set_xticklabels(names,rotation=30,ha="right")
        ax6[2].set_ylabel("WF RMS [nm]"); ax6[2].set_title("Wavefront RMS"); ax6[2].grid(True,alpha=0.3,axis='y')
        for i,v in enumerate(wfs): ax6[2].text(i,v+5,f"{v:.0f}",ha="center",fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig6.savefig(OUT/"phase2_fig6_metrics.png",dpi=150,bbox_inches="tight"); plt.close(fig6)
    print("  Saved fig6")

    del d0; torch.cuda.empty_cache()
    print(f"\nDone! Figures saved to {OUT}")

if __name__=="__main__":
    main()
