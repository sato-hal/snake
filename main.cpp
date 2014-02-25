// main.cpp
// 
//
//
//
aa
#include <vector>
using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;

#define GRAY_THRESH 130
#define ALPHA	1.0
#define BETA		1.0
#define GAMMA	1.2
#define D_TOTAL_THRESH		1.0
#define CURV_THRESH_FOR_BETA -0.7
#define GRAD_THRESH_FOR_BETA 20
#define ITERATION_MAX		1000
#define WAIT_INTERVAL 5

char *input_image=
	//"img/onepiece9-166.bmp"
	"watch_small.jpg"
;

Mat img;
Mat gray;
Mat binary;
Mat display;
Mat gradient;

bool is_button_down=false;
Point prev_pt;
vector<Point> point_seq;
vector<Point> point_seq_interval;

//---�X�l�[�N�J�n
void begin_snake(void)
{
	//---�_��̖��̂��ȒP�̂���"v"�ɕϊ�
	vector<Point> v;
	for(int i=0;i<(int)point_seq_interval.size();i++) v.push_back(point_seq_interval[i]);

	double d_total=DBL_MAX;
	double d_avg=0.0;
	int n=0;
	//double D_TOTAL_THRESH=1.0;
	//int ITERATION_MAX=500;
	int nei_loc[8][2]={ {1,0},{1,1},{0,1},{-1,1}, {-1,0},{-1,-1},{0,-1},{1,-1} };
	vector<double> beta_relaxed(v.size(),BETA);
	vector<double> curvature(v.size(),0.0);

	//---�f�o�b�O�p�o��
	cout<<"N         : "<<v.size()<<endl;

	//---������臒l�ȓ��Ȃ�J��Ԃ��_����ړ�
	while(d_total>=D_TOTAL_THRESH && n<=ITERATION_MAX){
		//---�ϐ��������E���ϒ��_�ԋ����̌v�Z
		n++;
		d_total=0.0;
		for(int i=0;i<(int)v.size();i++){
			int dx=v[(i+1)%v.size()].x-v[i].x;
			int dy=v[(i+1)%v.size()].y-v[i].y;
			d_avg += sqrt((double)(dx*dx+dy*dy));
		}
		d_avg/=v.size();
		//---�e���_v[i]�ɑ΂��G�l���M�[�̏����ȋߖT�_��T��
		for(int i=0;i<(int)v.size();i++){
			//---v[i]�ł̋Ǐ��G�l���M�[E_i�̌v�Z
			double I_max=0.0;
			double I_min=DBL_MAX;
			for(int j=0;j<8;j++){
				uchar intensity=gradient.at<uchar>(v[i].y+nei_loc[j][1],v[i].x+nei_loc[j][0]);
				if(intensity>I_max) I_max=intensity;
				if(intensity<I_min) I_min=intensity;
			}
			double E_image_i=-(gradient.at<uchar>(v[i].y,v[i].x)-I_min)/max(I_max-I_min,5.0);
			double E_cont_i=(d_avg-sqrt((double)(v[(i+1)%v.size()].x-v[i].x)*(v[(i+1)%v.size()].x-v[i].x)+(v[(i+1)%v.size()].y-v[i].y)*(v[(i+1)%v.size()].y-v[i].y)) )
				*(d_avg-sqrt((double)(v[(i+1)%v.size()].x-v[i].x)*(v[(i+1)%v.size()].x-v[i].x)+(v[(i+1)%v.size()].y-v[i].y)*(v[(i+1)%v.size()].y-v[i].y)) );
			double E_curv_i=(v[(i+1)%v.size()].x-2*v[i].x+v[(i-1)%v.size()].x)*(v[(i+1)%v.size()].x-2*v[i].x+v[(i-1)%v.size()].x)
				+(v[(i+1)%v.size()].y-2*v[i].y+v[(i-1)%v.size()].y)*(v[(i+1)%v.size()].y-2*v[i].y+v[(i-1)%v.size()].y);
			//E_cont_i/=I_max; //���K��
			//E_curv_i/=I_max;
			//if(I_max==0){
			//	E_cont_i=0.0;
			//	E_curv_i=0.0;
			//}
			double E_i=ALPHA*E_cont_i+beta_relaxed[i]*E_curv_i+GAMMA*E_image_i;
			//---v[i]��8�ߖTv_j(j=0,1,...,7)�ł̋Ǐ��G�l���M�[E_j�̌v�Z�A���̍ŏ��lE_j_min�����߂�
			Point v_j_min;
			double E_j_min=DBL_MAX;
			for(int j=0;j<8;j++){
				Point v_j=Point(v[i].x+nei_loc[j][0],v[i].y+nei_loc[j][1]); //��ʊO�ɏo���Ƃ��̗�O�������ĂȂ�
				double I_max_j=0.0;
				double I_min_j=DBL_MAX;
				for(int k=0;k<8;k++){
					uchar intensity_j=gradient.at<uchar>(v_j.y+nei_loc[k][1],v_j.x+nei_loc[k][0]);
					if(intensity_j>I_max_j) I_max_j=intensity_j;
					if(intensity_j<I_min_j) I_min_j=intensity_j;
				}
				double E_image_j=-(gradient.at<uchar>(v_j.y,v_j.x)-I_min_j)/max(I_max_j-I_min_j,5.0);
				double E_cont_j=(d_avg-sqrt((double)(v[(i+1)%v.size()].x-v_j.x)*(v[(i+1)%v.size()].x-v_j.x)+(v[(i+1)%v.size()].y-v_j.y)*(v[(i+1)%v.size()].y-v_j.y)) )
					*(d_avg-sqrt((double)(v[(i+1)%v.size()].x-v_j.x)*(v[(i+1)%v.size()].x-v_j.x)+(v[(i+1)%v.size()].y-v_j.y)*(v[(i+1)%v.size()].y-v_j.y)) );
				double E_curv_j=(v[(i+1)%v.size()].x-2*v_j.x+v[(i-1)%v.size()].x)*(v[(i+1)%v.size()].x-2*v_j.x+v[(i-1)%v.size()].x)
					+(v[(i+1)%v.size()].y-2*v_j.y+v[(i-1)%v.size()].y)*(v[(i+1)%v.size()].y-2*v_j.y+v[(i-1)%v.size()].y);
				//E_cont_j/=I_max_j; //���K��
				//E_curv_j/=I_max_j;
				//if(I_max_j==0){
				//	E_cont_j=0.0;
				//	E_curv_j=0.0;
				//}
				double E_j=ALPHA*E_cont_j+beta_relaxed[i]*E_curv_j+GAMMA*E_image_j;
				if(E_j<E_j_min){
					E_j_min=E_j;
					v_j_min=v_j;
				}
			}
			//---�Ǐ��G�l���M�[�ŏ��̋ߖT�_��v[i]���ړ�������E�ړ��ʂ����Z
			if(E_j_min<E_i){
				d_total+=sqrt( (double)((v[i].x-v_j_min.x)*(v[i].x-v_j_min.x)+(v[i].y-v_j_min.y)*(v[i].y-v_j_min.y)) );
				v[i]=v_j_min;
				//for debug
				//cout<<"E_i: "<<E_i<<"="<<E_cont_i<<", "<<E_curv_i<<", "<<E_image_i<<endl;
				//cout<<"E_j_min: "<<E_j_min<<endl<<endl;
			}
		}
		//---E_curv�̌W��BETA�̒����i�G�b�W�p�ł�BETA=0�Ɋɂ߂�j
		for(int i=0;i<(int)v.size();i++){
			Point u[2]={Point(v[i].x-v[(i-1)%v.size()].x,v[i].y-v[(i-1)%v.size()].y), Point(v[(i+1)%v.size()].x-v[i].x,v[(i+1)%v.size()].y-v[i].y)};
			if((u[0].x==0&&u[0].y==0) || (u[1].x==0&&u[1].y==0)) curvature[i]=2.0;
			else curvature[i]=(u[0].x/abs(u[0].x*u[0].x+u[0].y*u[0].y)-u[1].x/abs(u[1].x*u[1].x+u[1].y*u[1].y))*(u[0].x/abs(u[0].x*u[0].x+u[0].y*u[0].y)-u[1].x/abs(u[1].x*u[1].x+u[1].y*u[1].y))
				+(u[0].y/abs(u[0].x*u[0].x+u[0].y*u[0].y)-u[1].y/abs(u[1].x*u[1].x+u[1].y*u[1].y))*(u[0].y/abs(u[0].x*u[0].x+u[0].y*u[0].y)-u[1].y/abs(u[1].x*u[1].x+u[1].y*u[1].y));
		}
		for(int i=0;i<(int)v.size();i++){
			if(curvature[i]>curvature[(i-1)%v.size()] && curvature[i]>curvature[(i+1)%v.size()]
				&& curvature[i]>CURV_THRESH_FOR_BETA && gradient.at<uchar>(v[i].y,v[i].x)>GRAD_THRESH_FOR_BETA){ //curvature��0�`2�ɂȂ�
						beta_relaxed[i]=0.0;
			}
			else beta_relaxed[i]=BETA;
		}
		//---�\���p�摜�ɓ_��`��
		cvtColor(binary,display,CV_GRAY2BGR);
		for(int i=0;i<(int)v.size();i++){
			line(display,v[i],v[(i+1)%v.size()],Scalar(0,0,255),1);
			circle(display,v[i],2,Scalar(0,0,255),-1);
		}
		imshow("display",display);
		waitKey(WAIT_INTERVAL);
	}
	point_seq.clear();
	point_seq_interval.clear();
	v.clear();

	//---�f�o�b�O�p�o��
	cout<<"n         : "<<n<<endl;
	cout<<"d_total: "<<d_total<<endl;
}

//---�_���`��
void draw_point_seq(void)
{
	//---�_�̊Ԋu�𒲐�
	int MAX_INTERVAL=10;
	int MIN_INTERVAL=5;
	for(int i=0;i<(int)point_seq.size()-1;i++){
		double distance=sqrt( (double)((point_seq[i].x-point_seq[i+1].x)*(point_seq[i].x-point_seq[i+1].x)+(point_seq[i].y-point_seq[i+1].y)*(point_seq[i].y-point_seq[i+1].y)) );
		if(distance>MAX_INTERVAL){
			int added_pt_num=(int)floor(distance/MAX_INTERVAL);
			for(int j=0;j<added_pt_num;j++){
				point_seq_interval.push_back( Point(point_seq[i].x+(point_seq[i+1].x-point_seq[i].x)*j/added_pt_num, point_seq[i].y+(point_seq[i+1].y-point_seq[i].y)*j/added_pt_num) );
			}
		}
		else if(distance<MIN_INTERVAL){
			int j=i+2;
			while(j<=(int)point_seq.size()-1){
				double distance_j=sqrt( (double)((point_seq[i].x-point_seq[j].x)*(point_seq[i].x-point_seq[j].x)+(point_seq[i].y-point_seq[j].y)*(point_seq[i].y-point_seq[j].y)) );
				if(distance_j>MIN_INTERVAL){
					point_seq_interval.push_back(point_seq[j]);
					i=j;
					break;
				}
				else j++;
			}
		}
		else point_seq_interval.push_back(point_seq[i]);
	}
	//---�\���p�摜�ɓ_��`��
	cvtColor(binary,display,CV_GRAY2BGR);
	for(int i=0;i<(int)point_seq_interval.size();i++){
		line(display,point_seq_interval[i],point_seq_interval[(i+1)%point_seq_interval.size()],Scalar(0,0,255),1);
		circle(display,point_seq_interval[i],2,Scalar(0,0,255),-1);
	}
	imshow("display",display);
	//---�X�l�[�N�J�n
	begin_snake();
}

void on_mouse(int event, int x, int y, int flag, void*)
{
	if(event==CV_EVENT_MOUSEMOVE){
		if(flag&CV_EVENT_FLAG_LBUTTON){
			Point pt=Point(x,y);
			line(display,prev_pt,pt,Scalar(0,0,255),1);
			prev_pt=pt;
			imshow("display",display);
			//---�_��Ɍ��ݓ_��ǉ�
			point_seq.push_back(pt);
		}
	}
	else if(event==CV_EVENT_LBUTTONDOWN){
		if(point_seq_interval.size()==0){
			is_button_down=true;
			prev_pt=Point(x,y);
		}
	}
	else if(event==CV_EVENT_LBUTTONUP){
		is_button_down=false;
		//---�_���`��
		draw_point_seq();
	}
}

int main(int argc, char** argv)
{
	//---������摜�f�[�^�̊m��
	img=imread(input_image, 1); if(!img.data) return -1;
	gray=Mat(img.size(),CV_8UC1,Scalar(0));
	binary=gray.clone();
	display=Mat(img.size(),CV_8UC3,Scalar(0,0,0));

	//---�Q�l��
	cvtColor(img,gray,CV_BGR2GRAY);
	//threshold(gray,binary,GRAY_THRESH,255,THRESH_BINARY);
	binary=gray.clone();

	//---�O���f�B�G���g�v�Z
	Mat tmp_sobel;
	Sobel(gray,tmp_sobel,CV_32F,1,1);
	convertScaleAbs(tmp_sobel,gradient,1,0);
	imshow("gradient",gradient);

	//---�摜�̕\��
	cvtColor(binary,display,CV_GRAY2BGR);
	imshow("display",display);

	//---�}�E�X�R�[���o�b�N�֐��̓o�^
	setMouseCallback("display",on_mouse,0);

	//---�摜�̕ۑ�
	//imwrite("binary.png",binary);

	//---�L�[���͑҂�
	while(1){
		int key=waitKey(100);
		if(key=='c'){
			cvtColor(binary,display,CV_GRAY2BGR);
			imshow("display",display);
		}
		else if('a'<=key && key<='z') break;
	}

	return 0;
}
