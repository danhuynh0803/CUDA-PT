// (Attempting) pbr pt in CUDA using the Cook-Torrance model by Danny Huynh, 2017
// Based on smallptCUDA by Sam Lapere, 2015 

#include <iostream>
#include "gputimer.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "cutil_math.h"

#define M_PI 3.14159265359f  
#define width 640
#define height 480
#define samples 1024
#define alpha 0.5

// Other settings 
//#define CTBRDF   // Uncomment to use the Cook-Torrance reflectance model
//#define GLOBAL   // Uncomment to use only direct lighting


// ===============
// Related to direct lighting and shadowing 



// ===============


struct Ray { 
	float3 orig; 
	float3 dir; 
	__device__ Ray(float3 _orig, float3 _dir) : orig(_orig), dir(_dir) { } 
}; 

enum Refl_t { DIFF, SPEC, REFR }; 

struct Sphere { 
	float radius; 
	float3 pos, emis, albedo; 
	Refl_t refl; 

	__device__ float intersect_sphere(const Ray &r) const {
		//float3 oc = r.orig - pos;
		float3 oc = pos - r.orig;
		float t, epsilon = 0.0001f; 
		float b = dot(r.dir, oc); 
		float discr = b*b - dot(r.dir, r.dir) * (dot(oc, oc) - radius*radius); 
		//float discr = b*b - dot(oc, oc) + radius*radius;
		// if discriminant is not negative, then there is an intersection 
		if (discr < 0) { return 0; }
		else { discr = sqrtf(discr); }
		return (t = b - discr) > epsilon ? t : ((t = b + discr) > epsilon ? t : 0);
	}
}; 

// SCENE
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = 
{
	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.85f, 0.35f, 0.35f }, DIFF }, //Left
	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .35f, .35f, .85f }, DIFF}, //Rght
	{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back
	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt
	{ 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm
	{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top
	{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 0.9f, 0.9f, 0.8f }, DIFF }, // small sphere 1
	{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 0.1f, 0.3f, 1.0f }, DIFF }, // small sphere 2
	{ 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};


// check speeds if done inline
__device__ inline bool intersect_scene(const Ray &r, float &t, int &id) {
	float n = sizeof(spheres)/sizeof(Sphere);  
	float d, inf = t = FLT_MAX;
	
	for (int i = int(n); i >= 0; --i) { 
		if ( (d = spheres[i].intersect_sphere(r)) && d < t) { 
			t = d; 
			id = i;	
		}
	}
	return t < inf; 
}

// random number generator from https://github.com/gz/rust-raytracer
__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1) {
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	// Convert to float
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

	return (res.f - 2.f) / 2.f;
}

// For Diffuse Materials 
__device__ float3 uniform_sample_hemisphere(const float &r1, const float &r2) { 
	// r1 = cos(theta) = y 
	float sintheta = sqrtf(1 - r1*r1); 
	float phi = 2 * M_PI * r2;
   	float x = sintheta * cosf(phi); 	
	float z = sintheta * sinf(phi); 

	return make_float3(x, r1, z);
}

// For Specular Materials 
__device__ float3 reflect_sample_hemisphere(float3 light_dir, float3 norm)
{
	return light_dir - 2 * dot(light_dir, norm) * norm;
}


__device__ float reflect_coeff(float n1, float n2)
{
	float f0 = ( (n1 - n2) / (n1 + n2) );  
	return f0 * f0;  
}

// Calculates fresnel coefficient 
__device__ float fresnel(float3 l, float3 norm,  float n1, float n2)   
{
	float f0 = reflect_coeff(n1, n2); 
	return f0 + (1 - f0)*pow(1 - dot(l, norm), 5); 
}

// Calculates proportion of microfacets pointing in direction of half-vector h
__device__ float microfacet_dist(float3 m, float3 n)
{
	float cos_m = dot(m, n); 
	float tan_m = ( (1 - cos_m*cos_m) /cos_m ); 
	float numer = alpha * alpha * max(0.0f, dot(m, n));	
	// Distribution of microfacets is 
	float angle = (alpha * alpha) + (tan_m * tan_m); 
	float denom = M_PI * pow(cos_m, 4) * angle * angle;  

	return numer / denom;  
} 
// Calculates proportion of microfacets that are masked or shadowed  
__device__ float geometric_atten(float3 v, float3 l, float3 n)
{
	float3 h = normalize(v + l);  
	float view = (2 * max(0.0f, dot(n, h)) * max(0.0f, dot(n, v))) / max(0.0f, dot(v, h));
	float light = (2 * max(0.0f, dot(n, h)) * max(0.0f, dot(n, l))) / max(0.0f, dot(l, h));

	return min(1.0f, min(view, light));
}

// Compute the Cook-Torrance BRDF
__device__ float ct_brdf(const float3 norm, float3 &l, const float3 nl, unsigned int *s1, unsigned int *s2) 
{
	// Sample unit hemisphere 
	// create 2 random numbers
	float r1 = 2 * M_PI * getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
	float r2 = getrandom(s1, s2);  // pick random number for elevation
	float r2s = sqrtf(r2);

	float3 sampleLightDir = uniform_sample_hemisphere(r1, r2);	
	
	// Compute local orthonormal bases uvw at hitpoint to use for calculating random ray direction
	float3 w = nl;
	float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
	float3 v = cross(w, u);
	// Check which object our new light direction hits 
	 
			
	// get Fresnel 
	float F = fresnel(l, norm, 1.0f, 1.2f);  
	// get D 
	float D = microfacet_dist(l, norm);
	// get G 
	float G = geometric_atten(v, l, norm);
	
	float fr = (F * D * G) / (4 * dot(l, norm) * dot(v, norm));  
	
	// Set the sampled light direction as the new incident light direction
	l = sampleLightDir;	

	return fr;
}



// Radiance function, solves rendering equation
__device__ float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2)
{
	//("In radiance, at x:%d y:%d \n", *s1, *s2);
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	// ray bounce loop (no Russian Roulette) 

#ifndef GLOBAL
	accucolor = make_float3(1.0f, 1.0f, 1.0f);
	float3 shade = make_float3(0.1f, 0.1f, 0.1f);

	for (int bounces = 0; bounces < 1; ++bounces) {
		float t; 
		int id = 0; 
		if (!intersect_scene(r, t, id))
			return make_float3(0.0, 0.0f, 0.0f);

		const Sphere &hit = spheres[id];
		float3 p = r.orig + r.dir*t; 
		float3 n = normalize(p - hit.pos);
		//accucolor *= hit.albedo * max(dot(r.dir, n), 0.0f);
		float3 nl = dot(n, r.dir) < 0 ? n : n*-1; 
		if (id != 8) {
			accucolor *= hit.albedo * max(0.0f, dot(r.dir, -nl));
		}
		else {
			accucolor += hit.albedo;
		}
		Sphere light = spheres[8];
		float3 d = make_float3(light.pos.x, light.pos.y - light.radius, light.pos.z);
		r.orig = p + nl*0.05f; // offset ray origin slightly to prevent self intersection
		r.dir = normalize(d - p);

		// Shade area is the point is being blocked by another object
		if (intersect_scene(r, t, id)) { 
			if (id != 8) { 
				// Something blocking
				accucolor *= shade;
			}
			else {
				
			}
		} 	
	}


#else 
	int TotalBounces = 4; 

	for (int bounces = 0; bounces < TotalBounces; ++bounces) { 
		float t ; 
		int id = 0; 
		// if no intersection, then return black
		if (!intersect_scene(r, t, id)) 
			return make_float3(0.0f, 0.0f, 0.0f); 		
		// else, hit something!
		const Sphere &hit = spheres[id]; 
		float3 p = r.orig + r.dir*t; 
		float3 n = normalize(p - hit.pos); 
		float3 nl = dot(n, r.dir) < 0? n : n*-1;  // Flip normal if not facing camera

		// Add emission of current sphere to accumulated color 

		accucolor += mask * hit.emis;  // First term in rendering equation sum
		// create 2 random numbers
		float r1 = 2 * M_PI * getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
		float r2 = getrandom(s1, s2);  // pick random number for elevation
		float r2s = sqrtf(r2);

		// compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction
		// first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
		float3 w = nl;
		float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);

		float3 d; 
		float diff_coeff = 0.5;
		// Check object's material type
		if (hit.refl == DIFF) {
			// compute random ray direction on hemisphere using polar coordinates
			// cosine weighted importance sampling (favours ray directions closer to normal direction)
			d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
		}
		else if (hit.refl == SPEC) {
			d = reflect_sample_hemisphere(r.dir, n);
			diff_coeff = 0.2;
		}
		else if (hit.refl == REFR) {
			// TODO
		}
		else { // Default at diffuse
			d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
		}

		// new ray origin is intersection point of previous ray with scene
		r.orig = p + nl*0.05f; // offset ray origin slightly to prevent self intersection
		r.dir = d;

#ifdef CTBRDF
		// ============== 
		// CT-BRDF
		//
		float F = fresnel(r.dir, n, 1.0f, 1.2f);
		float D = microfacet_dist(r.dir, n);
		float G = geometric_atten(r.dir, d, n);
		float fr = (F * D * G) / (4 * dot(d, n) * dot(r.dir, n));
		
		mask *= diff_coeff * (hit.albedo * dot(d, nl)/M_PI) + (fr * (1.0 - diff_coeff)); 
		//mask *= hit.albedo * dot(d, nl) * (diff_coeff + fr * (1.0 - diff_coeff));
		mask *= 2;
		// ==============
#else 
		mask *= hit.albedo;    // multiply with colour of object
		mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
		mask *= 2;          // fudge factor
#endif // CTBRDF or not
	}

#endif // Direct vs Global Illumination
	return accucolor; 
}

__global__ void render_kernel(float3* output_d) 
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 

	unsigned int i = clamp((height - y - 1) * width + x, (unsigned int)0, (unsigned int) (width * height - 1));
	//printf("current pixel: %d\n", i);
	unsigned int s1 = x; 
	unsigned int s2 = y;

	float3 look_from = make_float3(50, 52, 295.6); 
	float3 look_at = normalize(make_float3(0, -0.042612, -1));

	// Set camera
	Ray cam(look_from, look_at);
	float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
	float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray direction offset in y direction (.5135 is FOV angle)
	float3 pixel_color = make_float3(0.0f);

	for (int s = 0; s < samples; ++s)
	{
		// Compute primary ray direction 
		float3 d = cam.dir + cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5);

		// Calculate the pixel color at the location 
		pixel_color = pixel_color + radiance(Ray(cam.orig + d * 40, normalize(d)), &s1, &s2) * (1.0 / samples);
		// Forced camera rays to be pushed forward to start in interior ^^
	}

	//printf("after radiance: %d\n", i);
	// Convert 2D to 1D
	output_d[i] = make_float3(clamp(pixel_color.x, 0.0f, 1.0f), clamp(pixel_color.y, 0.0f, 1.0f), clamp(pixel_color.z, 0.0f, 1.0f));

}

// Clamp values to be in range [0.0, 1.0]
inline float clamp(float x) { return x < 0.0f? 0.0f : x > 1.0f? 1.0f : x; }
// Converts RGB float in range [0, 1] to int range [0, 255], while performing gamma correction
inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); } 

int main()
{
	GpuTimer timer; 

	float3* output_h = new float3[3 * width * height]; 
	float3* output_d; 

	// allocate memory to gpu 
	cudaMalloc(&output_d, 3 * width * height * sizeof(float3));

	// specify the block and grid size for CUDA threads over SMs 
	dim3 block(16, 16, 1); 
	dim3 grid(width / block.x, height / block.y, 1); 

	timer.Start();
	// Launch 
	render_kernel <<< grid, block >>> (output_d); 
	cudaDeviceSynchronize();

	timer.Stop(); 

	cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	printf("GPU Processing Time: %g ms\n", timer.Elapsed());

	// Free any allocated memory on GPU
	cudaFree(output_d); 

	// Write to a ppm file 
	FILE *myFile = fopen("pt.ppm", "w"); 
	fprintf(myFile, "P3\n%d %d\n%d\n", width, height, 255); 
	for (int i = 0; i < width * height; ++i) 
	{ 
		fprintf(myFile, "%d %d %d ", toInt(output_h[i].x),
									 toInt(output_h[i].y), 
									 toInt(output_h[i].z));

	}

	// Free allocated memory on CPU
	delete[] output_h; 

	return 0;
}
