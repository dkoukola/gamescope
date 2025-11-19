#include "colorimetry.h"

#include "shaderfilter.h"
#include "alphamode.h"

vec4 sampleRegular(sampler2D tex, vec2 coord, uint colorspace) {
    vec4 color = textureLod(tex, coord, 0);
    color.rgb = colorspace_plane_degamma_tf(color.rgb, colorspace);
    return color;
}

// To be considered pseudo-bandlimited, upscaling factor must be at least 2x.

const float bandlimited_PI = 3.14159265359;
const float bandlimited_PI_half = 0.5 * bandlimited_PI;
// size: resolution of sampled texture
// inv_size: inverse resolution of sampled texture
// extent: Screen-space gradient of UV in texels. Typically computed as (texture resolution) / (viewport resolution).
//   If screen is rotated by 90 or 270 degrees, the derivatives need to be computed appropriately.
//   For uniform scaling, none of this matters.
//   extent can be multiplied to achieve LOD bias.
//   extent must be at least 1.0 / 256.0.
vec4 sampleBandLimited(sampler2D samp, vec2 uv, vec2 size, vec2 inv_size, vec2 extent, uint colorspace, bool unnormalized)
{
    // Josh:
    // Clamp to behaviour like 4x scale (0.25).
    //
    // Was defaulted to 2x before (0.5), which is 1px, but gives blurry result
    // on Cave Story (480p) -> 800p on Deck.
    // TODO: Maybe make this configurable?
    const float max_extent = 0.25f;

	// Get base pixel and phase, range [0, 1).
	vec2 pixel = uv * (unnormalized ? vec2(1.0f) : size) - 0.5;
	vec2 base_pixel = floor(pixel);
	vec2 phase = pixel - base_pixel;

	// We can resolve the filter by just sampling a single 2x2 block.
	// Lerp between normal sampling at LOD 0, and bandlimited pixel filter at LOD -1.
	vec2 shift = 0.5 + 0.5 * sin(bandlimited_PI_half * clamp((phase - 0.5) / min(extent, vec2(max_extent)), -1.0, 1.0));
	uv = (base_pixel + 0.5 + shift) * (unnormalized ? vec2(1.0f) : inv_size);

	return sampleRegular(samp, uv, colorspace);
}

uint pseudo_random(uint seed) {
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return seed * 1664525u + 1013904223u;
}

void compositing_debug(uvec2 coord) {
    uvec2 pos = coord;
    pos.x -= (u_frameId & 2) != 0 ?  128 : 0;
    pos.y -= (u_frameId & 1) != 0 ?  128 : 0;

    if (pos.x >= 40 && pos.x < 120 && pos.y >= 40 && pos.y < 120) {
        vec4 value = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        if (checkDebugFlag(compositedebug_Markers_Partial)) {
            value = vec4(0.0f, 1.0f, 1.0f, 1.0f);
        }
        if (pos.x >= 48 && pos.x < 112 && pos.y >= 48 && pos.y < 112) {
            uint random = pseudo_random(u_frameId.x + (pos.x & ~0x7) + (pos.y & ~0x7) * 50);
            vec4 time = round(unpackUnorm4x8(random)).xyzw;
            if (time.x + time.y + time.z + time.w < 2.0f)
                value = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        imageStore(dst, ivec2(coord), value);
    }
}

// Takes in a scRGB/Linear encoded value and applies color management
// based on the input colorspace.
//
// ie. call colorspace_plane_degamma_tf(color.rgb, colorspace) before
// input to this function.
vec3 apply_layer_color_mgmt(vec3 color, uint layer, uint colorspace) {
    if (colorspace == colorspace_passthru)
        return color;

    if (c_itm_enable)
    {
        color = bt2446a_inverse_tonemapping(color, u_itmSdrNits, u_itmTargetNits);
        colorspace = colorspace_pq;
    }

    // Shaper + 3D LUT path to match DRM.
    uint plane_eotf = colorspace_to_eotf(colorspace);

    if (layer == 0 && checkDebugFlag(compositedebug_Heatmap))
    {
        // Debug HDR heatmap.
        color = hdr_heatmap(color, colorspace);
        plane_eotf = EOTF_Gamma22;
    }


    // The shaper TF is basically just a regamma to get into something the shaper LUT can handle.
    //
    // Despite naming, degamma + shaper TF are NOT necessarily the inverse of each other. ^^^
    // This gets the color ready to go into the shaper LUT.
    // ie. scRGB -> PQ
    //
    // We also need to do degamma here for non-linear views to blend in linear space.
    // ie. PQ -> PQ would need us to manually do bilinear here.
    bool lut3d_enabled = textureQueryLevels(s_shaperLut[plane_eotf]) != 0;
    if (lut3d_enabled)
    {
        color = colorspace_plane_shaper_tf(color, colorspace);
        color = perform_1dlut(color, s_shaperLut[plane_eotf]);
        color = perform_3dlut(color, s_lut3D[plane_eotf]);
        color = colorspace_blend_tf(color, c_output_eotf);
    }

    return color;
}

vec4 sampleBilinear(sampler2D tex, vec2 coord, uint colorspace, bool unnormalized) {
    vec2 scale = unnormalized ? vec2(1.0) : vec2(textureSize(tex, 0));

    vec2 pixCoord = coord * scale - 0.5f;
    vec2 originPixCoord = floor(pixCoord);

    vec2 gatherUV = (originPixCoord * scale + 1.0f) / scale;

    vec4 red   = textureGather(tex, gatherUV, 0);
    vec4 green = textureGather(tex, gatherUV, 1);
    vec4 blue  = textureGather(tex, gatherUV, 2);
    vec4 alpha = textureGather(tex, gatherUV, 3);

    vec4 c00 = vec4(red.w, green.w, blue.w, alpha.w);
    vec4 c01 = vec4(red.x, green.x, blue.x, alpha.x);
    vec4 c11 = vec4(red.y, green.y, blue.y, alpha.y);
    vec4 c10 = vec4(red.z, green.z, blue.z, alpha.z);

    c00.rgb = colorspace_plane_degamma_tf(c00.rgb, colorspace);
    c01.rgb = colorspace_plane_degamma_tf(c01.rgb, colorspace);
    c11.rgb = colorspace_plane_degamma_tf(c11.rgb, colorspace);
    c10.rgb = colorspace_plane_degamma_tf(c10.rgb, colorspace);

    vec2 filterWeight = pixCoord - originPixCoord;

    vec4 temp0 = mix(c01, c11, filterWeight.x);
    vec4 temp1 = mix(c00, c10, filterWeight.x);
    return mix(temp1, temp0, filterWeight.y);
}

const float subpixelKernelR[49] = float[49](
    -1.0526e-02, -5.1071e-02, -8.5119e-02, -6.1192e-02, -9.1344e-03,  3.5897e-03,  5.5279e-03,
    -2.4249e-03,  2.6682e-02,  2.3826e-02,  2.4192e-02,  1.0538e-02,  1.9939e-03, -1.1452e-03,
    -1.9276e-03,  1.1129e-01,  1.5322e-01,  1.1598e-01,  2.3158e-02, -2.1166e-02, -1.9158e-02,
    -9.8290e-03,  1.7032e-01,  2.3965e-01,  1.8491e-01,  2.8468e-02, -4.0587e-02, -4.3002e-02,
    -2.0463e-03,  1.1078e-01,  1.5392e-01,  1.1533e-01,  2.3271e-02, -2.1582e-02, -1.7926e-02,
    -2.5467e-03,  2.6298e-02,  2.4420e-02,  2.3553e-02,  1.0961e-02,  1.5187e-03, -1.6591e-04,
    -1.0415e-02, -5.1437e-02, -8.4871e-02, -6.2043e-02, -8.7220e-03,  3.4067e-03,  6.0927e-03
);

const float subpixelKernelG[49] = float[49](
     3.8514e-03, -5.4484e-03, -5.0128e-02, -8.9105e-02, -5.3422e-02, -6.5463e-03,  4.3577e-03,
    -4.5839e-03,  1.0168e-02,  2.2712e-02,  2.5182e-02,  2.1549e-02,  1.0217e-02, -3.9492e-03,
    -3.3225e-02,  1.9505e-02,  1.0794e-01,  1.6799e-01,  1.0714e-01,  1.9900e-02, -3.2187e-02,
    -5.6365e-02,  2.1324e-02,  1.6300e-01,  2.5932e-01,  1.6246e-01,  2.2935e-02, -5.5869e-02,
    -3.3211e-02,  1.9467e-02,  1.0804e-01,  1.6802e-01,  1.0719e-01,  1.9986e-02, -3.2075e-02,
    -4.5877e-03,  1.0157e-02,  2.2687e-02,  2.5167e-02,  2.1636e-02,  1.0232e-02, -3.8778e-03,
     3.8308e-03, -5.4800e-03, -5.0085e-02, -8.9107e-02, -5.3422e-02, -6.5015e-03,  4.4328e-03
);

const float subpixelKernelB[49] = float[49](
     1.1707e-02, -1.5588e-02, -2.6335e-02, -5.8017e-02, -5.6514e-02, -3.8170e-02, -2.2234e-02,
    -2.0525e-02,  2.2323e-02, -5.5703e-03,  4.2159e-02,  1.9901e-03,  3.5535e-02, -1.5025e-02,
    -1.5280e-02, -4.1840e-02,  2.8163e-02,  1.2661e-01,  1.7142e-01,  1.0636e-01, -3.2487e-03,
    -4.0351e-02,  5.7782e-03,  5.7683e-02,  1.6544e-01,  1.8679e-01,  1.5421e-01,  1.2911e-02,
    -4.0655e-03, -3.8520e-02,  4.1863e-02,  9.2533e-02,  1.6531e-01,  1.0219e-01,  1.5415e-02,
    -1.0176e-02,  2.1141e-02,  9.2225e-03,  1.2310e-02,  1.4885e-03,  3.3014e-02, -4.7787e-04,
     9.5637e-03, -2.3571e-02, -2.7867e-02, -4.9899e-02, -4.5404e-02, -3.7778e-02, -3.0901e-02
);

vec4 sampleSubpixelDownscaleRGBVertical(sampler2D tex, vec2 pixelCoord, vec2 texSize, vec2 scale, uint colorspace)
{
    // Filter expects a 3:1 downscale ratio. Fall back to bilinear if we aren't close enough.
    if (any(greaterThan(abs(scale - vec2(3.0f)), vec2(0.01f))))
    {
        return sampleBilinear(tex, pixelCoord, colorspace, true);
    }

    ivec2 origin = ivec2(round(pixelCoord)) - ivec2(2);
    ivec2 bounds = ivec2(texSize) - ivec2(1);

    vec3 accum = vec3(0.0f);
    float accumA = 0.0f;

    for (int ky = 0; ky < 7; ky++)
    {
        for (int kx = 0; kx < 7; kx++)
        {
            ivec2 sampleCoord = origin + ivec2(kx, ky);
            sampleCoord = clamp(sampleCoord, ivec2(0), bounds);

            vec4 texel = texelFetch(tex, sampleCoord, 0);
            vec3 linear = colorspace_plane_degamma_tf(texel.rgb, colorspace);

            int kernelIndex = ky * 7 + kx;
            accum.r += linear.r * subpixelKernelR[kernelIndex];
            accum.g += linear.g * subpixelKernelG[kernelIndex];
            accum.b += linear.b * subpixelKernelB[kernelIndex];

            float weightA = (subpixelKernelR[kernelIndex] + subpixelKernelG[kernelIndex] + subpixelKernelB[kernelIndex]) * (1.0f / 3.0f);
            accumA += texel.a * weightA;
        }
    }

    return vec4(accum, accumA);
}

vec4 sampleLayerEx(sampler2D layerSampler, uint offsetLayerIdx, uint colorspaceLayerIdx, vec2 uv, bool unnormalized) {
    vec2 coord = ((uv + u_offset[offsetLayerIdx]) * u_scale[offsetLayerIdx]);
    vec2 texSize = textureSize(layerSampler, 0);

    if (coord.x < 0.0f       || coord.y < 0.0f ||
        coord.x >= texSize.x || coord.y >= texSize.y) {
        float border = (u_borderMask & (1u << offsetLayerIdx)) != 0 ? 1.0f : 0.0f;

        if (checkDebugFlag(compositedebug_PlaneBorders))
            return vec4(vec3(1.0f, 0.0f, 1.0f) * border, border);

        return vec4(0.0f, 0.0f, 0.0f, border);
    }

    vec2 pixelCoord = coord;

    if (!unnormalized)
        coord /= texSize;

    uint colorspace = get_layer_colorspace(colorspaceLayerIdx);
    vec4 color;
    if (get_layer_shaderfilter(offsetLayerIdx) == filter_pixel) {
        vec2 output_res = texSize / u_scale[offsetLayerIdx];
        vec2 extent = max((texSize / output_res), vec2(1.0 / 256.0));
        color = sampleBandLimited(layerSampler, coord, unnormalized ? vec2(1.0f) : texSize, unnormalized ? vec2(1.0f) : vec2(1.0f) / texSize, extent, colorspace, unnormalized);
    }
    else if (get_layer_shaderfilter(offsetLayerIdx) == filter_subpixel_rgb_vertical) {
        color = sampleSubpixelDownscaleRGBVertical(layerSampler, pixelCoord, texSize, u_scale[offsetLayerIdx], colorspace);
    }
    else if (get_layer_shaderfilter(offsetLayerIdx) == filter_linear_emulated) {
        color = sampleBilinear(layerSampler, coord, colorspace, unnormalized);
    }
    else {
        color = sampleRegular(layerSampler, coord, colorspace);
    }
    // JoshA: AMDGPU applies 3x4 CTM like this, where A is 1.0, but it only affects .rgb.
    color.rgb = vec4(color.rgb, 1.0f) * u_ctm[colorspaceLayerIdx];
    color.rgb = apply_layer_color_mgmt(color.rgb, offsetLayerIdx, colorspace);

    return color;
}

vec4 sampleLayer(sampler2D layerSampler, uint layerIdx, vec2 uv, bool unnormalized) {
    return sampleLayerEx(layerSampler, layerIdx, layerIdx, uv, unnormalized);
}

vec3 encodeOutputColor(vec3 value) {
    return colorspace_output_tf(value, c_output_eotf);
}
