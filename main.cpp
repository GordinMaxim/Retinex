#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Retinex.hpp"

int main() {
    cv::String folderpath = "/home/maxim/Videos/campus/campus4-c1_img/scene00*.png";
    std::vector<cv::String> filenames;
    cv::glob(folderpath, filenames);

    for (const auto &filename : filenames) {
        auto img = cv::imread(filename);
//        auto ssr = retinex::singleScaleRetinex(img, 80);
//        auto msr = retinex::multiScaleRetinex(img, {15, 80, 250});
        retinex::colorRestoration(img, 0, 0);
//        cv::imshow("ssr", ssr);
//        cv::imshow("msr", msr);
                cv::waitKey(1);
        cv::imshow("image", img);
    }
    return 0;
}