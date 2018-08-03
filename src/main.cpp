#include <utils.h>

CAFFE2_DEFINE_string(init_net, "res/squeezenet_init_net.pb", "the given path to the init protobuffer.")
CAFFE2_DEFINE_string(predict_net, "res/squeezenet_predict_net.pb", "predict protobuffer");
CAFFE2_DEFINE_string(file, "res/test.jpg", "input image file");
CAFFE2_DEFINE_string(classes, "res/imagenet_classes.txt", "the classes file.");
CAFFE2_DEFINE_int(size, 227, "the image file.")

namespace caffe2 {

    void print(const Blob *blob, const std::string &name) {
        const auto tensor = blob->Get<TensorCUDA>().Clone();
        const auto &data = tensor.data<float>();
        std::cout << name << "(" << tensor.dims() << "):" << std::vector<float>(data, data + tensor.size())
                  << std::endl;
    }

    TensorCPU prepareMatImgData(cv::Mat &image) {
        cv::Size scale(std::max(FLAGS_size * image.cols / image.rows, FLAGS_size),
                       std::max(FLAGS_size, FLAGS_size * image.rows / image.cols));
        cv::resize(image, image, scale);
        std::cout << "scaled size:" << image.size() << std::endl;

        cv::Rect crop((image.cols - FLAGS_size) / 2, (image.rows - FLAGS_size) / 2, FLAGS_size, FLAGS_size);
        image = image(crop);
        std::cout << "cropped size:" << image.size() << std::endl;

        image.convertTo(image, CV_32FC3, 1.0, -128);

        vector<cv::Mat> channels(3);
        cv::split(image, channels);
        std::vector<float> data;
        for (auto &c :channels) {
            data.insert(data.end(), (float *) c.datastart, (float *) c.dataend);
        }
        std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});
        TensorCPU tensor_host = TensorCPU(dims, data, NULL);
        return tensor_host;
    }

    void caffe2_pretrained_run() {
        if (!std::ifstream(FLAGS_init_net).good() ||
            !std::ifstream(FLAGS_predict_net).good()) {
            std::cerr << "model file missiong" << std::endl;
            return;
        }

        if (!std::ifstream(FLAGS_file).good()) {
            std::cerr << "error, image missing" << std::endl;
            return;
        }

        std::cout << "start ..." << std::endl;
        std::cout << "init net ..." << std::endl;

        //try gpu
        auto bflag = HasCudaGPU();
        bflag = HasCudaRuntime();
#ifdef __GPU__
        DeviceOption option;
        option.set_device_type(CUDA);
        new CUDAContext(option);
#endif

        NetDef init_net, predict_net;
        CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
        CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

        //transfer net to gpu
#ifdef __GPU__
        init_net.mutable_device_option()->set_device_type(CUDA);
        predict_net.mutable_device_option()->set_device_type(CUDA);
#endif
        Workspace workspace("tmp");
        CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
#ifdef __GPU__
        auto input = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
#else
        auto input = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
#endif

        std::cout << "load classes..." << std::endl;

        std::ifstream file(FLAGS_classes);
        std::string temp;
        std::vector<std::string> classes;
        while (std::getline(file, temp)) {
            classes.push_back(temp);
        }

        std::cout << "init net done ..." << std::endl;

        std::cout << "init camera ..." << std::endl;
        cv::VideoCapture cap("res/fruits.mp4");
        /*if(!cap.isOpened()){
            std::cout<<"camera open failed..."<<std::endl;
            return;
        }*/
        std::cout << "camera done..." << std::endl;
        cv::Mat o_image;
        auto &image = o_image;//= cv::imread(FLAGS_file);
        for (int i = 0; i < 10000; i++) {
            auto cap_result = cap.read(image);
            auto show_image = image.clone();
            std::cout << "cap result is:" << cap_result << std::endl;
            std::cout << "image size:" << image.size() << std::endl;

            TensorCPU tensor_host = prepareMatImgData(image);
            input->CopyFrom(tensor_host);

            CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));

            auto &output_name = predict_net.external_output(0);
#ifdef __GPU__
            auto output_device = workspace.GetBlob(output_name)->Get<TensorCUDA>().Clone();
#else
            auto output_device = workspace.GetBlob(output_name)->Get<TensorCPU>().Clone();
#endif
            //transfer GPU tensor to CPU tensor, must do it, or it will crash
#ifdef __GPU__
            auto output = TensorCPU(output_device);
#else
            auto &output = output_device;
#endif
            const auto &probs = output.data<float>();
            std::vector<std::pair<int, int>> pairs;
            for (auto i = 0; i < output.size(); i++) {
                if (probs[i] > 0.1) {
                    pairs.push_back(std::make_pair(probs[i] * 100, i));
                }
            }

            std::sort(pairs.begin(), pairs.end());
            std::cout << std::endl;

            std::cout << "output:" << std::endl;
            for (auto pair:pairs) {
                std::cout << " " << pair.first << "% " << classes[pair.second] << std::endl;
            }
            cv::imshow("test", show_image);
            cv::waitKey(1);
        }
    }
}//namespace caffe2_first

int main(int argc, char **argv) {
    caffe2::GlobalInit(&argc, &argv);
    caffe2::caffe2_pretrained_run();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
