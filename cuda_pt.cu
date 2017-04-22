// (Attempting) pbr pt in CUDA by Danny Huynh, 2017
// Based on smallptCUDA by Sam Lapere, 2015 

#include <iostream>
#include "gputimer.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "cutil_math.h"

#define M_PI 3.14159265359f  
#define width 512
#define height 384
#define samps 1024

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
		float3 oc = r.orig - pos; 
		float t, epsilon = 0.0001f; 
		float b = dot(r.dir, oc); 
		float discr = b*b - dot(r.dir, r.dir) * (dot(oc, oc) - radius*radius); 
		// if discriminant is not negative, then there is an intersection 
		if (discr < 0) { return; }
		else { discr = sqrtf(discr); }
		return (t = b - discr) > epsilon ? t : ((t = b + discr) > epsilon ? t : 0);
	}
}; 

// SCENE
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = {

};

// check speeds if done inline
__device__ bool intersect_scene(const Ray &r, float &t, int &id) {
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

__device__ float3 uniform_sample_hemisphere(const float &r1, const float &r2) { 
	// r1 = cos(theta) = y 
	float sintheta = sqrtf(1 - r1*r1); 
	float phi = 2 * M_PI * r2;
   	float x = sintheta * cosf(phi); 	
	float z = sintheta * sinf(phi); 

	return make_float3(x, r1, z);
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
__device__ float microfacet_dist(float3 m, float3 n, float alpha) 
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
	float h = norm(v + l);  
	float view = (2 * dot(n, h) * dot(n, v)) / dot(v,h); 
	float light = (2 * dot(n, h) * dot(n, l)) / dot(l, h);

	return min(1, view, light);
}

// Compute the Cook-Torrance BRDF
__device__ float ct_brdf() 
{

}



// Radiance function, solves rendering equations 
__device__ float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2) { 
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	// ray bounce loop (no Russian Roulette) 
	for (int bounces = 0; bounces < 4; ++bounces) { 
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
		


	}
}

__global__ void render_kernel(float3* output_d) 
{

}

// Clamp values to be in range [0.0, 1.0]
inline float clamp(float x) { return x < 0.0f? 0.0f : x > 1.0f? 1.0f : x; }
// Converts RGB float in range [0, 1] to int range [0, 255], while performing gamma correction
inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); } 

int main()
{
	GpuTimer timer; 

	float3* output_h = new float3[width * height]; 
	float3* output_d; 

	// allocate memory to gpu 
	cudaMalloc(&output_d, width * height * sizeof(float3));

	// specify the block and grid size for CUDA threads over SMs 
	dim3 block(8, 8, 1); 
	dim3 grid(width / block.x, height / block.y, 1); 

	timer.Start();
	// Launch 
	render_kernel <<< grid, block >>> (output_d); 
	timer.Stop(); 

	cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	printf("GPU Processing Time: %g ms\n", timer.Elapsed());

	// Free any allocated memory on GPU
	cudaFree(output_d); 

	// Write to a ppm file 
	FILE *myFile = fopen("pt.ppm", "w"); 
	fprintf(myFile, "P3\n%d %d\n %d\n", width, height, 255); 
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
