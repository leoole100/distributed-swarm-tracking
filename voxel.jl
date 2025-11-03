using GLMakie

function project_point(pt::AbstractVector{<:Real}, P::AbstractMatrix{<:Real})
    X = Float64.([pt[1], pt[2], pt[3], 1.0])
    uvw = P * X
    w = uvw[3]
    return (uvw[1] / w, uvw[2] / w)
end

fx, fy = 800.0, 800.0
cx, cy = 320.0, 240.0
P = Float64[
    fx 0  cx 0;
    0  fy cy 0;
    0  0  1  0
]

width, height = 640, 480
img = zeros(Float32, (height, width))

s = 10
cx = width ÷ 2
cy = height ÷ 2
img[
    cy - s : cy + s,
    cx - s : cx + s,
] .= 1.0f0

# create a voxel grid in front of the camera and loop over coordinates
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
zmin, zmax = 0.1, 1.0

nx, ny, nz = 101, 101, 101
xs = range(xmin, xmax, length=nx)
ys = range(ymin, ymax, length=ny)
zs = range(zmin, zmax, length=nz)

vox = zeros(Float32, (nx, ny, nz))

function sample_image(img::AbstractArray{T,2}, u::Real, v::Real) where T
    h, w = size(img)
    iu = floor(Int, u)   # column index (x)
    iv = floor(Int, v)   # row index (y)
    if iu < 1 || iv < 1 || iu > w || iv > h
        return zero(T)
    else
        return img[iv, iu]
    end
end

# project the images into the voxel grid
function voxelize!(vox, xs, ys, zs, P, img)
    nx, ny, nz = size(vox)
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        x = xs[ix]; y = ys[iy]; z = zs[iz]
        u, v = project_point([x, y, z], P)
        vox[ix, iy, iz] = sample_image(img, u, v)
    end
    return nothing
end

# warm-up to trigger compilation
voxelize!(vox, xs, ys, zs, P, img)

# timed run (no compilation)
t0 = time()
voxelize!(vox, xs, ys, zs, P, img)
t_elapsed = time() - t0
println("Voxelization time (no compilation): $(t_elapsed) seconds")

volume(
    # xs, ys, zs,
    vox, 
    interpolate=false,
    colormap=to_colormap([RGBAf(0,0,0,0), RGBAf(0,0,0,1)])
)
