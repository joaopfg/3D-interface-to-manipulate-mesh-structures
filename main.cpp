#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <ostream>
#include <igl/readOFF.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>

#include <igl/gaussian_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h> 

#include "HalfedgeBuilder.cpp"  

#include <bits/stdc++.h>

using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;

const double eps = 1e-9;

typedef pair<int,int> ii;

MatrixXd V;
MatrixXi F;


MatrixXd N_faces;   //computed calling pre-defined functions of LibiGL
MatrixXd N_vertices; //computed calling pre-defined functions of LibiGL


MatrixXd lib_N_vertices;  //computed using face-vertex structure of LibiGL
MatrixXi lib_Deg_vertices;//computed using face-vertex structure of LibiGL

MatrixXd he_N_vertices; //computed using the HalfEdge data structure

struct Point{
	double x, y, z;
	Point() {}
	Point(double x, double y, double z): x(x), y(y), z(z) {}
};

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
	switch(key){
		case '1':
			viewer.data().set_normals(N_faces);
			return true;
		case '2':
			viewer.data().set_normals(N_vertices);
			return true;
		case '3':
			viewer.data().set_normals(lib_N_vertices);
			return true;
		case '4':
			viewer.data().set_normals(he_N_vertices);
			return true;
		default: break;
	}
	return false;
}

//Return the degree of a given vertex 'v'
int vertexDegree(HalfedgeDS he, int v) {
	int deg = 0, eCrawl = he.getEdge(v);

	while(true){
		deg++;
		eCrawl = he.getOpposite(eCrawl);
		eCrawl = he.getPrev(eCrawl);

		if(eCrawl == he.getEdge(v)) break;
	}

	return deg;
}

 double norm(Point p){
 	return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
 }

 Point cross_prod(Point a, Point b, Point c){
 	Point ab = Point(b.x - a.x, b.y - a.y, b.z - a.z);
 	Point ac = Point(c.x - a.x, c.y - a.y, c.z - a.z);

 	return Point((ab.y*ac.z - ab.z*ac.y)/2.0, (ac.x*ab.z  - ac.z*ab.x)/2.0, (ab.x*ac.y - ab.y*ac.x)/2.0);
 }

 double dot_prod(Point a, Point b, Point c){
 	Point ab = Point(b.x - a.x, b.y - a.y, b.z - a.z);
 	Point ac = Point(c.x - a.x, c.y - a.y, c.z - a.z);

 	return ab.x*ac.x + ab.y*ac.y + ab.z*ac.z;
 }

//Compute the vertex normals (he)
void vertexNormals(HalfedgeDS he) {
	cout << endl;
	cout << "Computing vertex normals using half-edge data structure..." << endl;
	auto start = std::chrono::high_resolution_clock::now();

	he_N_vertices = MatrixXd::Zero(he.sizeOfVertices(), 3);

	bool visit[he.sizeOfHalfedges()];
	memset(visit,false,sizeof(visit));

	for(int i=0;i<he.sizeOfHalfedges();i++){
		if(!visit[i]){
			int edges[3], vertex[3];

			edges[0] = i; 
			edges[1] = he.getNext(edges[0]);
			edges[2]= he.getPrev(edges[0]);

			for(int j=0;j<3;j++){
				visit[edges[j]] = true;
				vertex[j] = he.getTarget(edges[j]);
			}

			Point p[3];

			for(int j=0;j<3;j++){
				MatrixXd cur = V.row(vertex[j]);
				p[j] = Point(cur(0,0), cur(0,1), cur(0,2));
			}

			Point face_normal = cross_prod(p[0], p[1], p[2]);

			for(int j=0;j<3;j++){
				he_N_vertices(vertex[j], 0) += face_normal.x;
				he_N_vertices(vertex[j], 1) += face_normal.y;
				he_N_vertices(vertex[j], 2) += face_normal.z;
			}
		}
	}

	//Normalizing all the vectors
	for(int i=0;i<he.sizeOfVertices();i++){
		Vector3d v(he_N_vertices(i,0), he_N_vertices(i,1), he_N_vertices(i,2));
		v.normalize();
		he_N_vertices(i,0) = v(0,0);
		he_N_vertices(i,1) = v(1,0);
		he_N_vertices(i,2) = v(2,0);
	}

	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Running time: " << elapsed.count() << " s\n";
    cout << endl;
}

//Compute lib_per-vertex normals
//Compute the vertex normals (global, using libiGl data structure)
void lib_vertexNormals() {
	cout << endl;
	cout << "Computing vertex normals using libigl data structure..." << endl;
	auto start = std::chrono::high_resolution_clock::now();

	lib_N_vertices = MatrixXd::Zero(V.rows(), 3);

	Point lib_N_faces[F.rows()];

	for(int i=0;i<F.rows();i++){
		Point p[3];

		for(int j=0;j<3;j++){
			MatrixXd cur = V.row(F(i,j));
			p[j] = Point(cur(0,0), cur(0,1), cur(0,2));
		}

		lib_N_faces[i] = cross_prod(p[0], p[1], p[2]);
	}

	for(int i=0;i<F.rows();i++){
		Point p = lib_N_faces[i];
		double x = p.x, y = p.y, z = p.z;

		for(int j=0;j<3;j++){
			lib_N_vertices(F(i,j), 0) += x;
			lib_N_vertices(F(i,j), 1) += y;
			lib_N_vertices(F(i,j), 2) += z;
		}
	}

	//Normalizing all the vectors
	for(int i=0;i<V.rows();i++){
		Vector3d v(lib_N_vertices(i,0), lib_N_vertices(i,1), lib_N_vertices(i,2));
		v.normalize();
		lib_N_vertices(i,0) = v(0,0);
		lib_N_vertices(i,1) = v(1,0);
		lib_N_vertices(i,2) = v(2,0);
	}

	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Running time: " << elapsed.count() << " s\n";
    cout << endl;
}

//Return the number of occurrence of vertex degrees: for d=3..n-1

void vertexDegreeStatistics(HalfedgeDS he) {
	cout << endl;	 	
	cout << "Computing vertex degree distribution using half-edge data structure..." << endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	int deg[he.sizeOfVertices()];
	memset(deg,0,sizeof(deg));

	for(int i=0;i<he.sizeOfVertices();i++) deg[vertexDegree(he, i)]++;

	for(int i=3;i<=13;i++) cout << "deg = " << i << ": " << deg[i] << endl;

	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Running time: " << elapsed.count() << " s\n";
    cout << endl;
}

//Compute lib_vertex degrees
//(global, using libiGl data structure)
//Exercice

void lib_vertexDegrees() {
	cout << endl;
	cout << "Computing vertex degree distribution using libigl data structure..." << endl;
	auto start = std::chrono::high_resolution_clock::now();
	map<ii, bool> visit;
	int deg[V.rows()], quantDeg[V.rows()];
	memset(deg,0,sizeof(deg));
	memset(quantDeg,0,sizeof(quantDeg));

	for(int i=0;i<F.rows();i++){
		if(visit.count({F(i,0), F(i,1)}) == 0){
			visit[{F(i,0), F(i,1)}] = true;
			deg[F(i,0)]++;
		}
		if(visit.count({F(i,0), F(i,2)}) == 0){
			visit[{F(i,0), F(i,2)}] = true;
			deg[F(i,0)]++;
		}
		if(visit.count({F(i,1), F(i,0)}) == 0){
			visit[{F(i,1), F(i,0)}] = true;
			deg[F(i,1)]++;
		}
		if(visit.count({F(i,1), F(i,2)}) == 0){
			visit[{F(i,1), F(i,2)}] = true;
			deg[F(i,1)]++;
		}
		if(visit.count({F(i,2), F(i,0)}) == 0){
			visit[{F(i,2), F(i,0)}] = true;
			deg[F(i,2)]++;
		}
		if(visit.count({F(i,2), F(i,1)}) == 0){
			visit[{F(i,2), F(i,1)}] = true;
			deg[F(i,2)]++;
		}
	}

	for(int i=0;i<V.rows();i++) quantDeg[deg[i]]++;

	for(int i=3;i<=13;i++) cout << "deg = " << i << ": " << quantDeg[i] << endl;

	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Running time: " << elapsed.count() << " s\n";
    cout << endl;
}
         
//Return the number of boundaries of the mesh
int countBoundaries(HalfedgeDS he){
	cout << endl;
	cout << "Counting boundaries on the half-edge data structure..." << endl;
	auto start = std::chrono::high_resolution_clock::now();

	bool visit[he.sizeOfHalfedges()];
	memset(visit,false,sizeof(visit));

	int boundaries = 0;

	for(int i=0;i<he.sizeOfHalfedges();i++){
		if(!visit[i]){
			int e = i;
			e = he.getNext(e);
			e = he.getNext(e);
			e = he.getNext(e);

			if(e != i){
				visit[i] = true;

				e = i;
				e = he.getNext(e);

				while(e != i){
					visit[e] = true;
					e = he.getNext(e);
				}

				boundaries++;
			}
		}
	}

	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Running time: " << elapsed.count() << " s\n";
    cout << endl;

	return boundaries;
}  

void lib_Gaussian_curvature(VectorXd & K){
	double PI = acos(-1.0);

	for(int i=0;i<V.rows();i++) K(i) = 2.0*PI;

	for(int i=0;i<F.rows();i++){
		Point p[3];

		for(int j=0;j<3;j++){
			MatrixXd cur = V.row(F(i,j));
			p[j] = Point(cur(0,0), cur(0,1), cur(0,2));
		}

		for(int j=0;j<3;j++){
			if(j == 0){
				double dot = dot_prod(p[0],p[1],p[2]);

				Point p0p1 = Point(p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z);
				Point p0p2 = Point(p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z);
				
				K(F(i,j)) -= acos(dot/norm(p0p1)/norm(p0p2));
			}
			else if(j == 1){
				double dot = dot_prod(p[1],p[2],p[0]);

				Point p1p0 = Point(p[0].x - p[1].x, p[0].y - p[1].y, p[0].z - p[1].z);
				Point p1p2 = Point(p[2].x - p[1].x, p[2].y - p[1].y, p[2].z - p[1].z);
				
				K(F(i,j)) -= acos(dot/norm(p1p0)/norm(p1p2));
			}
			else{
				double dot = dot_prod(p[2],p[0],p[1]);

				Point p2p1 = Point(p[1].x - p[2].x, p[1].y - p[2].y, p[1].z - p[2].z);
				Point p2p0 = Point(p[0].x - p[2].x, p[0].y - p[2].y, p[0].z - p[2].z);
				
				K(F(i,j)) -= acos(dot/norm(p2p1)/norm(p2p0));
			}
		}
	}
}

void Half_edge_Gaussian_curvature(HalfedgeDS he, VectorXd & K){
	double PI = acos(-1.0);

	for(int i=0;i<V.rows();i++) K(i) = 2.0*PI;

	for(int i=0;i<he.sizeOfVertices();i++){		
		Point p[3];

		p[0] = Point(V(i,0), V(i,1), V(i,2));

		int e = he.getEdge(i);

		while(true){
			e = he.getNext(e);
			p[1] = Point(V(he.getTarget(e), 0), V(he.getTarget(e), 1), V(he.getTarget(e), 2));
			e = he.getNext(e);
			p[2] = Point(V(he.getTarget(e), 0), V(he.getTarget(e), 1), V(he.getTarget(e), 2));
			e = he.getNext(e);

			double dot = dot_prod(p[0],p[1],p[2]);

			Point p0p1 = Point(p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z);
			Point p0p2 = Point(p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z);
			if(norm(p0p1) > eps && norm(p0p2) > eps) K(i) -= acos(dot/norm(p0p1)/norm(p0p2));

			e = he.getOpposite(e);
			e = he.getPrev(e);
			if(e == he.getEdge(i)) break;
		}
	}
}

//Can't pass by reference here because the K values will be modified
void Gaussian_curvature(VectorXd K, HalfedgeDS he, int method){
	if(method == 0){
		cout << endl;
		cout << "Calculating gaussian curvature using function from libigl..." << endl;
		auto start = std::chrono::high_resolution_clock::now();

		igl::gaussian_curvature(V,F,K);

		auto finish = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> elapsed = finish - start;
    	std::cout << "Running time: " << elapsed.count() << " s\n";
    	cout << endl;
	}
	else if(method == 1){
		cout << endl;
		cout << "Calculating gaussian curvature using face vertex data structure..." << endl;
		auto start = std::chrono::high_resolution_clock::now();

		lib_Gaussian_curvature(K);

		auto finish = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> elapsed = finish - start;
    	std::cout << "Running time: " << elapsed.count() << " s\n";
    	cout << endl;	
	}
	else if(method == 2){
		cout << endl;
		cout << "Calculating gaussian curvature using half-edge data structure..." << endl;
		auto start = std::chrono::high_resolution_clock::now();

		Half_edge_Gaussian_curvature(he, K);

		auto finish = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> elapsed = finish - start;
    	std::cout << "Running time: " << elapsed.count() << " s\n";
    	cout << endl;	
	}
	else{
		cout << endl;
		cout << "Method to compute gaussian curvature not found" << endl;
		cout << endl;
		return;
	} 

	SparseMatrix<double> M,Minv;
  	igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
  	igl::invert_diag(M,Minv);
  	K = (Minv*K).eval();

  	MatrixXd C;
  	igl::jet(K,true,C);

  	// Plot the mesh with pseudocolors
  	igl::opengl::glfw::Viewer viewer;
  	viewer.data().set_mesh(V, F);
  	viewer.data().set_colors(C);
  	viewer.launch();
}

// ------------ main program ----------------
int main(int argc, char *argv[]) {
	igl::readOFF("../data/cube_open.off",V,F);

	//print the number of mesh elements
    cout << "Points: " << V.rows() << std::endl;

    HalfedgeBuilder* builder=new HalfedgeBuilder();  //

    HalfedgeDS he=builder->createMesh(V.rows(), F);  //

	// compute vertex degrees
    vertexDegreeStatistics(he); 
    lib_vertexDegrees();

	// compute number of boundaries
    int B = countBoundaries(he);  
    if(B == 1) cout << "The mesh has " << B << " boundary" << std::endl;
    else cout << "The mesh has " << B << " boundaries" << std::endl;

	// Compute normals

	// Compute per-face normals
	igl::per_face_normals(V,F,N_faces);


	// Compute per-vertex normals
	igl::per_vertex_normals(V,F,N_vertices);


	// Compute lib_per-vertex normals
	lib_vertexNormals();


	// Compute he_per-vertex normals
    vertexNormals(he);  //

    //Compute gaussian curvature with libigl data structure
    VectorXd K(V.rows());
    Gaussian_curvature(K, he, 0);
    Gaussian_curvature(K, he, 1);
    Gaussian_curvature(K, he, 2);
 ///////////////////////////////////////////

  // Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

 	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.data().set_mesh(V, F);  //
	viewer.data().set_normals(N_faces);  //
	std::cout<<
	"Press '1' for per-face normals calling pre-defined functions of LibiGL." << std::endl <<
	"Press '2' for per-vertex normals calling pre-defined functions of LibiGL." << std::endl <<
    "Press '3' for lib_per-vertex normals using face-vertex structure of LibiGL ." << std::endl <<
	"Press '4' for HE_per-vertex normals using HalfEdge structure." << std::endl;


	VectorXd Z;
	Z.setZero(V.rows(),1);

   // Use the z coordinate as a scalar field over the surface

	Z = V.col(2);

	MatrixXd C;
  // Assign per-vertex colors
	igl::jet(Z,true,C);
	viewer.data().set_colors(C);  // Add per-vertex colors

  //viewer.core(0).align_camera_center(V, F);  //not needed
	viewer.launch(); // run the viewer
}
