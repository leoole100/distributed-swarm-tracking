# pip install moderngl vedo numpy

import numpy as np
import moderngl
from vedo import Volume, Plotter, Axes, Box

# ---- GPU compute shader for back-projection ----
COMPUTE_SRC = r"""
#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;
layout(std430, binding=0) buffer Vol { float data[]; };

uniform int nx, ny, nz;
uniform float x0, y0, z0;
uniform float sx, sy, sz;

uniform mat3 R_cw;
uniform vec3 t_cw;

uniform float fx, fy, cx, cy;
uniform int W, H;

layout(binding = 0) uniform sampler2D img;

uint lin_index(int ix, int iy, int iz) {
    return uint((ix * ny + iy) * nz + iz);
}

void main() {
    int ix = int(gl_GlobalInvocationID.x);
    int iy = int(gl_GlobalInvocationID.y);
    int iz = int(gl_GlobalInvocationID.z);
    if (ix >= nx || iy >= ny || iz >= nz) return;

    float xw = x0 + (float(ix) + 0.5) * sx;
    float yw = y0 + (float(iy) + 0.5) * sy;
    float zw = z0 + (float(iz) + 0.5) * sz;

    vec3 Xc = R_cw * vec3(xw, yw, zw) + t_cw;
    if (Xc.z <= 1e-6) { data[lin_index(ix,iy,iz)] = 0.0; return; }

    float u = fx * (Xc.x / Xc.z) + cx;
    float v = fy * (Xc.y / Xc.z) + cy;
    if (u < 0.0 || u > float(W-1) || v < 0.0 || v > float(H-1)) {
        data[lin_index(ix,iy,iz)] = 0.0; return;
    }

    vec2 uv = vec2((floor(u)+0.5)/float(W), (floor(v)+0.5)/float(H));
    float val = texture(img, uv).r;
    data[lin_index(ix,iy,iz)] = val;
}
"""

def compute_volume(nx=128, ny=128, nz=128, W=320, H=180):
    # --- synthetic image: bright disk ---
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((xx-W*0.6)**2 + (yy-H*0.55)**2)
    img_np = (r < min(W,H)*0.2).astype(np.float32)

    # --- camera pose (look from -x to center) ---
    cx_w, cy_w, cz_w = nx/2, ny/2, nz/2
    cam_pos = np.array([-150.0, cy_w, cz_w], np.float32)
    look_at = np.array([cx_w, cy_w, cz_w], np.float32)
    up      = np.array([0,1,0], np.float32)

    zc = (look_at - cam_pos); zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up); xc /= np.linalg.norm(xc)
    yc = np.cross(xc, zc)
    R_wc = np.stack([xc, yc, zc], axis=1).astype(np.float32)
    R_cw = R_wc.T
    t_cw = (-R_cw @ cam_pos).astype(np.float32)

    # intrinsics
    fx=fy=700.0; cx=W/2; cy=H/2

    # grid mapping
    x0=y0=z0=0.0; sx=sy=sz=1.0

    # --- GL compute ---
    ctx = moderngl.create_standalone_context(require=430)
    prog = ctx.compute_shader(COMPUTE_SRC)

    tex = ctx.texture((W,H), 1, img_np.tobytes(), dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.use(0)

    nvox = nx*ny*nz
    vol_ssbo = ctx.buffer(reserve=nvox*4)
    vol_ssbo.bind_to_storage_buffer(0)

    # uniforms
    for name,val in dict(nx=nx,ny=ny,nz=nz,
                         x0=x0,y0=y0,z0=z0,
                         sx=sx,sy=sy,sz=sz,
                         fx=float(fx),fy=float(fy),cx=float(cx),cy=float(cy),
                         W=W,H=H).items():
        prog[name].value = val
    prog['R_cw'].write(R_cw.tobytes())
    prog['t_cw'].write(t_cw.tobytes())

    # dispatch
    ldx,ldy,ldz=8,8,4
    gx=(nx+ldx-1)//ldx
    gy=(ny+ldy-1)//ldy
    gz=(nz+ldz-1)//ldz
    prog.run(group_x=gx, group_y=gy, group_z=gz)
    ctx.finish()

    vol = np.frombuffer(vol_ssbo.read(), dtype=np.float32).reshape(nx,ny,nz)
    return vol

# ---- main ----
if __name__=="__main__":
    vol = compute_volume()

    nx,ny,nz = vol.shape
    vol_actor = Volume(vol, spacing=(1,1,1), origin=(0,0,0))
    vmin,vmax = float(vol.min()), float(vol.max())
    p1,p2,p3 = np.percentile(vol, [70,90,99])
    vol_actor.cmap("viridis", vmin=vmin, vmax=vmax)
    vol_actor.alpha([(vmin,0.0),(p1,0.0),(p2,0.1),(p3,0.4),(vmax,1.0)])
    vol_actor.mode(1)

    bbox = vol_actor.box().c("white").alpha(0.2)   # <- fixed
    axes = Axes(vol_actor, xygrid=True)

    Plotter(axes=axes).show(vol_actor, bbox).close()
