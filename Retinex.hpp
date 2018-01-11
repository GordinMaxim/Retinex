#pragma once

#include <numeric>
#include <map>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace retinex {
    auto colorRestoration(const cv::Mat &img, double alpha, double beta) {
        cv::Mat mat, img_sum;
        img.convertTo(mat, CV_32F);
        cv::transform(img, img_sum, cv::Matx13f(1,1,1));
        img_sum.convertTo(img_sum, CV_32F);
        mat *= alpha;
        cv::log(mat, mat);
        cv::log(img_sum, img_sum);
        auto den = log(10);

        cv::Mat out;
        cv::Mat in[] = {img_sum, img_sum, img_sum};
        cv::merge(in, 3, out);

        mat -= out;
        mat /= den;
        mat *= beta;
        return mat;
    }

    /**
     * Count unique channel values by pair
     * @param img
     * @return channelValue -> count
     */
    auto uniqueColors(const cv::Mat& img) {
        std::unordered_map<uchar, size_t> table;
        auto nRows = img.rows;
        auto nCols = img.cols;
        if (img.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }
        const uchar* p;
        for(auto i = 0; i < nRows; ++i) {
            p = img.ptr<uchar>(i);
            for (auto j = 0; j < nCols; ++j) {
                table[p[j]] += 1;
            }
        }
        return table;
    }

    auto simplestColorBalance(const cv::Mat &img, double low_clip, double high_clip) {
        auto total = img.total();
        std::vector<cv::Mat> channelMatrices(img.channels());
        cv::split(img, channelMatrices);
        for (auto& channelMat : channelMatrices) {
            auto table = uniqueColors(channelMat);
            double current = 0.;
            uchar low_val = 0;
            uchar high_val = 0;
            for (auto& p : table) {
                if (current / total < low_clip)
                    low_val = p.first;
                if (current / total < high_clip)
                    high_val = p.first;
                current += p.second;
            }
            auto ingore_val = 255;
            /**
             * assume always high_val > low_val
             */
//            cv::threshold(channelMat, channelMat, high_val, ingore_val, cv::THRESH_TRUNC);
            std::transform(channelMat.data, channelMat.data + channelMat.total(), channelMat.data,
                           [low_val, high_val] (uchar c) {
                               if (high_val < c)
                                   return high_val;
                               if (c > low_val)
                                   return c;
                               return low_val;
                           });
        }
        cv::Mat mat;
        cv::merge(channelMatrices, mat);
        return mat;
    }

    auto singleScaleRetinex(const cv::Mat &img, double sigma) {
        cv::Mat mat;
        img.convertTo(mat, CV_32F);
        cv::Mat gauss(img.size(), img.type());
        cv::GaussianBlur(img, gauss, cv::Size(0, 0), sigma);
        gauss.convertTo(gauss, CV_32F);

        /**
         * retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
         * &
         * lg - log_10
         * ln - log_e
         * lg(a) = ln(a) / ln(10)
         */
        cv::log(mat, mat);
        cv::log(gauss, gauss);
        auto den = log(10);
        mat -= gauss;
        mat /= den;

//        int channels = mat.channels();
//        int nRows = mat.rows;
//        int nCols = mat.cols * channels;
//
//        if (mat.isContinuous()) {
//            nCols *= nRows;
//            nRows = 1;
//        }
//
//        int i,j;
//        double* p;
//        double* pg;
//        for(i = 0; i < nRows; ++i) {
//            p = mat.ptr<double>(i);
//            pg = gauss.ptr<double>(i);
//            for (j = 0; j < nCols; ++j) {
//                p[j] = log10(p[j]) - log10(pg[j]);
//            }
//        }
        return mat;
    }

    auto multiScaleRetinex(const cv::Mat &img, const std::vector<double> &sigmas) {
        auto start = singleScaleRetinex(img, sigmas[0]);
        std::vector<decltype(start)> matrices;
        for (size_t i = 1; i < sigmas.size(); ++i) {
            matrices.emplace_back(singleScaleRetinex(img, sigmas[i]));
        }
        auto mat = std::accumulate(matrices.begin(), matrices.end(), start);
        mat /= sigmas.size();

        return mat;
    }

    auto MSRCR(const cv::Mat &img,
               const std::vector<double> &sigmas,
               double G, double b, double alpha, double beta,
               double low_clip, double high_clip) {
//        img = np.float64(img) + 1.0
        auto img_retinex = multiScaleRetinex(img, sigmas);
        auto img_color = colorRestoration(img, alpha, beta);

        cv::Mat img_msrcr = G * (img_retinex * img_color + b);
//        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        cv::Mat img_double;
        img_msrcr.convertTo(img_double, CV_64F);
        auto ptr = img_double.ptr<double>();
        std::transform(ptr, ptr + img_msrcr.total(), ptr,
                       [] (double c) {
                           if (255. < c)
                               return 255.;
                           if (0. < c)
                               return c;
                           return 0.;
                       });
        img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip);

        return img_msrcr;
    }

    auto automatedMSRCRR(const cv::Mat &img, const std::vector<double> &sigmas) {
        auto img_retinex = multiScaleRetinex(img + 1.0, sigmas);

        return img_retinex;
    }

    auto MSRCP(const cv::Mat &img, const std::vector<double> &sigmas, double low, double high) {

    }
}
