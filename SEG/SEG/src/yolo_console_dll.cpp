#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>
//#include <io.h>
#include <dirent.h>
//#include <windows.h>
//#include <WinInet.h>
#include <map>
#include <ctime>
#include "json.h"
#include "json-forwards.h"

#include "yolo_v2_class.hpp"    // imported functions from DLL

#ifdef OPENCV

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH     // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
#endif    // TRACK_OPTFLOW
#endif    // USE_CMAKE_LIBS
#else     // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH

using namespace std;
vector<string> m_ListTotalFileName;
map<string, vector<string>> xray_mapset;
vector<string> xray_type_name = { "X1_FSAP", "X2_FSLL", "X2_FSLR", "X3_FLOL", "X3_FLOR", "X4_HAV", "X5_AWBAP", "X6_AWBLL", "X6_AWBLR", "X7_KWBAP", "X8_KWBLL", "X8_KWBLR", "T1_TG" };

#endif    // OPENCV


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};

vector<string> split(string str, char Delimiter) {
	istringstream iss(str);             // istringstream에 str을 담는다.
	string buffer;                      // 구분자를 기준으로 절삭된 문자열이 담겨지는 버퍼

	vector<string> result;

	// istringstream은 istream을 상속받으므로 getline을 사용할 수 있다.
	while (getline(iss, buffer, Delimiter)) {
		result.push_back(buffer);               // 절삭된 문자열을 vector에 저장
	}

	return result;
}

/*
int isFileOrDir(_finddata_t fd)
{
	if (fd.attrib & _A_SUBDIR)
		return 0;
	else
		return 1;
}

void searchingDir(string path)
{
	int checkDirFile = 0;
	string dirPath = path + "\\*.*";
	struct _finddata_t fd;
	intptr_t handle;


	if ((handle = _findfirst(dirPath.c_str(), &fd)) == -1L)
		return;
	do
	{
		checkDirFile = isFileOrDir(fd);
		if (checkDirFile == 0 && fd.name[0] != '.') {
			//cout << "Dir  : " << path << "\\" << fd.name << endl;
			searchingDir(path + "\\" + fd.name);
		}
		else if (checkDirFile == 1 && fd.name[0] != '.') {
			std::string str_file(fd.name);
			std::string const file_ext = str_file.substr(str_file.find_last_of(".") + 1);
			if (file_ext == "png" || file_ext == "PNG")
			{
				string _file = path + "\\" + fd.name;
				m_ListTotalFileName.push_back(_file);
			}
		}

	} while (_findnext(handle, &fd) == 0);
	_findclose(handle);
}
*/
void searchingDir()
{
    struct dirent *de;  // Pointer for directory entry
 
    // opendir() returns a pointer of DIR type. 
    DIR *dr = opendir("./eval/png");
 
    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory" );
    }
 
    // for readdir()
    while ((de = readdir(dr)) != NULL) 
    {
        if (de->d_type == DT_REG){
                string str1(de->d_name);
                string test = "./eval/png/" + str1;
                m_ListTotalFileName.push_back(test);
        }
    }
 
    closedir(dr);
}


int main(int argc, char *argv[])
{
  std::ofstream syslog_File("system_log.csv");
  struct tm curr_tm;
  time_t curr_time = time(nullptr);
  localtime_r(&curr_time, &curr_tm);
  int curr_year = curr_tm.tm_year + 1900;
  int curr_month = curr_tm.tm_mon + 1;
  int curr_day = curr_tm.tm_mday;
  int curr_hour = curr_tm.tm_hour;
  int curr_minute = curr_tm.tm_min;
  int curr_second = curr_tm.tm_sec;
  string cur_time = to_string(curr_year) + "-" + to_string(curr_month) + "-" + to_string(curr_day) + "-" + to_string(curr_hour) + ":" + to_string(curr_minute) + ":" + to_string(curr_second);
  syslog_File << "Program Start Time\n";
  syslog_File << cur_time + "\n";
  syslog_File << "Command \n";
  std::string str_cmd(argv[1]);
  syslog_File << "LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib " + str_cmd + "\n";
    
	
	system("rm -rf eval");
	int nResult_2 = remove("evaluation_result.csv");
	int nResult_3 = remove("inspection.csv");
	int nResult_4 = remove("result.csv");
	int nResult_5 = remove("seg.csv");
	
	// xray dataset selecting
	std::cout << "Searching x-ray dataset" << std::endl;
	std::string dataset_full_path;
	if (argc > 1) dataset_full_path = argv[1];
	else {
		std::cout << "input dataset path" << std::endl;
		return 0;
	}

	if (dataset_full_path.size() == 0)
		std::cin >> dataset_full_path;
	std::cout << dataset_full_path << std::endl;

	std::string cmd_1 = "python3 image_seleting.py " + dataset_full_path;
	system(cmd_1.c_str());
	// end
	
	// classification model upload
	std::cout << "uploading Deep Learning Model" << std::endl;
	std::string  names_file = "./data/obj.names";
	std::string  cfg_file = "./data/yolov4.cfg";
	std::string  weights_file = "./data/yolov4_last.weights";
	Detector detector(cfg_file, weights_file);
	// end

	// start
	std::ofstream inspection_File("inspection.csv");
	inspection_File << "Class_ID,Object_Inspection\n";
	inspection_File.close();
	std::ofstream seg_File("seg.csv");
	seg_File << "Class_ID,Overlapping_Region,Combined_Region,Object_IoU,Mean_IoU\n";
	seg_File.close();
	//std::ofstream eval_File("evaluation_result.csv");
	//eval_File << "Class_ID\n";
	searchingDir();
	for (size_t i = 0; i < m_ListTotalFileName.size(); i++)
	{
		std::cout << m_ListTotalFileName[i] << std::endl;
		// checking x-ray type
		std::cout << "checking x-ray type" << std::endl;
		cv::Mat xray_img = cv::imread(m_ListTotalFileName[i]);

		if (!xray_img.empty())
		{
			///cv::imshow("test", xray_img);
			///cv::waitKey(1);
			auto det_image = detector.mat_to_image_resize(xray_img);

			std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, xray_img.size().width, xray_img.size().height);
			std::string classify_xray_name = "";
			for (auto &i : result_vec) {
				classify_xray_name = xray_type_name[i.obj_id];
			}

			std::string base = m_ListTotalFileName[i].substr(m_ListTotalFileName[i].find_last_of("/") + 1);
			vector<string> base_name = split(base, '.');

			if (!classify_xray_name.empty()) {
				// person label check
				//eval_File << base_name[0] << ","; // file name
				std::string xray_json_file = "./eval/json/" + base_name[0] + ".json";
				std::string errorMessage;
				Json::Value root;
				Json::CharReaderBuilder reader;
				ifstream is(xray_json_file, ifstream::binary);
				auto bret = Json::parseFromStream(reader, is, &root, &errorMessage);
				if (bret == false) {
					cout << "Error to parse JSON file" << endl;
				}
				auto nodeList = root["ArrayOfannotation_info"];
				bool xray_type_flag = true;
				for (auto n : nodeList) {
					auto xyvList = n["xyvalue"];
					auto labelList = xyvList["label_val"];
					auto label_type_name = labelList["preset_name"];

					if (label_type_name != classify_xray_name) {
						xray_type_flag = false;
						classify_xray_name = label_type_name.toStyledString();
						classify_xray_name.erase(std::find(classify_xray_name.begin(), classify_xray_name.end(), '\n'));
						classify_xray_name.erase(std::find(classify_xray_name.begin(), classify_xray_name.end(), '\"'));
						classify_xray_name.erase(std::find(classify_xray_name.begin(), classify_xray_name.end(), '\"'));
					}
				}
				if (xray_type_flag) {
					//eval_File << classify_xray_name << ": True";
					std::cout << classify_xray_name << ": True" << std::endl;
				}
				else {
					//eval_File << classify_xray_name << ": False";
					std::cout << classify_xray_name << ": False" << std::endl;
				}

				xray_mapset[classify_xray_name].push_back(base_name[0]);

				// checking x-ray segmentation
				//std::cout << "checking x-ray segmentation" << std::endl;
				//std::string cmd_3 = "python .\\SOLO\\demo\\seg_detect.py " + classify_xray_name + " " + m_ListTotalFileName[i] + " " + xray_json_file;
				//system(cmd_3.c_str());

			}
			else {
				std::cout << "dont detect x-ray type" << std::endl;
				//eval_File << base_name[0] << ","; // file name
				//eval_File << "Dont Detect,";
			}
			// end
			//eval_File << "\n";
		}
		else
		{
			std::cout << "dicom file error: " << m_ListTotalFileName[i] << std::endl;
		}
		
	}

	//eval_File.close();
	
	
	
	
	
	for (size_t z = 0; z < xray_type_name.size(); z++)
	{
		std::ofstream filename_File("filelist.csv", std::ofstream::trunc);
		std::cout << xray_type_name[z] << std::endl;
		string file_list_str = "";
		for (size_t j = 0; j < xray_mapset[xray_type_name[z]].size(); j++)
		{
			filename_File << xray_mapset[xray_type_name[z]][j] << ",";
		}
		filename_File.close();
		//std::cout << file_list_str << std::endl;
		std::cout << "checking x-ray segmentation" << std::endl;
		std::string cmd_3 = "python3 ./SOLO/demo/seg_detect.py " + xray_type_name[z] + " filelist.csv";
		system(cmd_3.c_str());
	}


	//std::string cmd_4 = "python test.py";
	//system(cmd_4.c_str());
  
  struct tm end_tm;
  time_t endd_time = time(nullptr);
  localtime_r(&endd_time, &end_tm);
  int end_year = end_tm.tm_year + 1900;
  int end_month = end_tm.tm_mon + 1;
  int end_day = end_tm.tm_mday;
  int end_hour = end_tm.tm_hour;
  int end_minute = end_tm.tm_min;
  int end_second = end_tm.tm_sec;
  string end_time = to_string(end_year) + "-" + to_string(end_month) + "-" + to_string(end_day) + "-" + to_string(end_hour) + ":" + to_string(end_minute) + ":" + to_string(end_second);
  syslog_File << "Program End Time\n";
  syslog_File << end_time + "\n";
  syslog_File.close();
	
    return 0;
}
