#include <iostream>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>

#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

CAFFE2_DEFINE_string(init_net, "res/squeezenet_init_net.pb","the given path to the init protobuffer.")
CAFFE2_DEFINE_string(predict_net,"res/squeezenet_predict_net.pb","predict protobuffer");
CAFFE2_DEFINE_string(file,"res/test.jpg","input image file");
CAFFE2_DEFINE_string(classes,"res/imagenet_classes.txt","the classes file.");
CAFFE2_DEFINE_int(size,227,"the image file.")



namespace caffe2{

    void print(const Blob* blob, const std::string& name){
        auto tensor = blob->GetMutable<TensorCPU>().Clone();
        const auto& data = tensor.data<float>();
        std::cout<< name << "(" << tensor.dims() << "):" <<std::vector<float>(data,data+tensor.size()) << std::endl;
    }

    void caffe2_1st_run(){
        std::cout << std::endl;
        std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
        std::cout << std::endl;

        Workspace workspace;
        std::vector<float> x(4*3*2);
        for(auto& v : x){
            v = (float)rand()/RAND_MAX;
        }

        std::cout << x << std::endl;
        {
            std::cout << "feed blob my_x" << std::endl;
            auto tensor = workspace.CreateBlob("my_x")->GetMutable<TensorCPU>();
            auto value = TensorCPU({4,3,2},x,NULL);
            tensor->ResizeLike(value);
            tensor->ShareData(value);
        }

        {
            const auto blob = workspace.GetBlob("my_x");
            print(blob,"my_x");
        }

        std::vector<float> data(16*100);
        for(auto& v:data){
            v = (float)rand()/RAND_MAX;
        }

        std::vector<int> label(16);
        for(auto& v:label){
            v = 10*rand()/RAND_MAX;
        }

        {
            auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
            auto value = TensorCPU({16,100},data,NULL);
            tensor->ResizeLike(value);
            tensor->ShareData(value);
        }

        {
            auto tensor = workspace.CreateBlob("label")->GetMutable<TensorCPU>();
            auto value = TensorCPU({16},label,NULL);
            tensor->ResizeLike(value);
            tensor->ShareData(value);
        }

        NetDef initModel;
        initModel.set_name("my first net_init");
        NetDef predictModel;
        predictModel.set_name("my first net");

        {
            auto op = initModel.add_op();
            op->set_type("XavierFill");
            auto arg = op->add_arg();
            arg->set_name("shape");
            arg->add_ints(10);
            arg->add_ints(100);
            op->add_output("fc_w");
        }

        {
            auto op = initModel.add_op();
            op->set_type("ConstantFill");
            auto arg = op->add_arg();
            arg->set_name("shape");
            arg->add_ints(10);
            op->add_output("fc_b");
        }

        std::vector<OperatorDef*> gradient_ops;
        {
            auto op = predictModel.add_op();
            op->set_type("FC");
            op->add_input("data");
            op->add_input("fc_w");
            op->add_input("fc_b");
            op->add_output("fc1");
            gradient_ops.push_back(op);
        }

        {
            auto op = predictModel.add_op();
            op->set_type("Sigmoid");
            op->add_input("fc1");
            op->add_output("pred");
            gradient_ops.push_back(op);
        }

        {
            auto op = predictModel.add_op();
            op->set_type("SoftmaxWithLoss");
            op->add_input("pred");
            op->add_input("label");
            op->add_output("softmax");
            op->add_output("loss");
            gradient_ops.push_back(op);
        }

        {
            auto op = predictModel.add_op();
            op->set_type("ConstantFill");
            auto arg = op->add_arg();
            arg->set_name("value");
            arg->set_f(1.0);
            op->add_input("loss");
            op->add_output("loss_grad");
            op->set_is_gradient_op(true);
        }

        std::reverse(gradient_ops.begin(),gradient_ops.end());
        for(auto op:gradient_ops){
            vector<GradientWrapper> output(op->output_size());
            for(auto i = 0; i<output.size();i++){
                output[i].dense_ = op->output(i)+"_grad";
            }
            GradientOpsMeta meta = GetGradientForOp(*op,output);
            auto grad = predictModel.add_op();
            grad->CopyFrom(meta.ops_[0]);
            grad->set_is_gradient_op(true);
        }

        CAFFE_ENFORCE(workspace.RunNetOnce(initModel));
        CAFFE_ENFORCE(workspace.CreateNet(predictModel));

        for(auto i = 0; i < 100; i++){
            std::vector<float> data(16*100);
            for (auto& v:data){
                v = (float)rand()/RAND_MAX;
            }

            std::vector<int> label(16);
            for(auto& v:label){
                v = 10*rand()/RAND_MAX;
            }

            {
                auto tensor = workspace.GetBlob("data")->GetMutable<TensorCPU>();
                auto value = TensorCPU({16,100},data,NULL);
                tensor->ShareData(value);
            }

            {
                auto tensor = workspace.GetBlob("label")->GetMutable<TensorCPU>();
                auto value = TensorCPU({16},label,NULL);
                tensor->ShareData(value);
            }

            for (auto j = 0; j<100;j++){
                CAFFE_ENFORCE(workspace.RunNet(predictModel.name()));
            }
        }

        std::cout<<std::endl;
        print(workspace.GetBlob("softmax"),"softmax");
        std::cout<<std::endl;
        print(workspace.GetBlob("loss"),"loss");

    }
    void caffe2_pretrained_run(){
        if(!std::ifstream(FLAGS_init_net).good() ||
           !std::ifstream(FLAGS_predict_net).good()){
            std::cerr << "model file missiong" << std::endl;
            return;
        }

        if(!std::ifstream(FLAGS_file).good()){
            std::cerr<<"error, image missing"<<std::endl;
            return;
        }

        std::cout<<"start ..."<<std::endl;
        std::cout<<"init net ..." <<std::endl;

        NetDef init_net, predict_net;
        CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net,&init_net));
        CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net,&predict_net));

        Workspace workspace("tmp");
        CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
        auto input = workspace.CreateBlob("data")->GetMutable<TensorCPU>();

        std::cout<<"load classes..." << std::endl;

        std::ifstream file(FLAGS_classes);
        std::string temp;
        std::vector<std::string> classes;
        while(std::getline(file,temp)){
            classes.push_back(temp);
        }

        std::cout<<"init net done ..." <<std::endl;

        std::cout<<"init camera ..." << std::endl;
        cv::VideoCapture cap("res/fruits.mp4");
        /*if(!cap.isOpened()){
            std::cout<<"camera open failed..."<<std::endl;
            return;
        }*/
        std::cout<<"camera done..." <<std::endl;
        cv::Mat o_image;
        auto& image = o_image ;//= cv::imread(FLAGS_file);
        for(int i = 0; i < 10000; i++)
        {
            auto cap_result = cap.read(image);
            auto show_image = image.clone();
            std::cout << "cap result is:" << cap_result << std::endl;
            std::cout << "image size:" << image.size() << std::endl;
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
            TensorCPU tensor(dims, data, NULL);

            /*NetDef init_net, predict_net;
            CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net,&init_net));
            CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net,&predict_net));

            Workspace workspace("tmp");
            CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
            auto input = workspace.CreateBlob("data")->GetMutable<TensorCPU>();*/
            input->ResizeLike(tensor);
            input->ShareData(tensor);
            CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));

            auto &output_name = predict_net.external_output(0);
            auto output = workspace.GetBlob(output_name)->GetMutable<TensorCPU>();

            const auto &probs = output->data<float>();
            std::vector<std::pair<int, int>> pairs;
            for (auto i = 0; i < output->size(); i++) {
                if (probs[i] > 0.1) {
                    pairs.push_back(std::make_pair(probs[i] * 100, i));
                }
            }

            std::sort(pairs.begin(), pairs.end());
            std::cout << std::endl;

            /*std::ifstream file(FLAGS_classes);
            std::string temp;
            std::vector<std::string> classes;
            while(std::getline(file,temp)){
                classes.push_back(temp);
            }*/

            std::cout << "output:" << std::endl;
            for (auto pair:pairs) {
                std::cout << " " << pair.first << "% " << classes[pair.second] << std::endl;
            }
            cv::imshow("test",show_image);
            cv::waitKey(1);
        }
    }
}//namespace caffe2_first

int main(int argc, char** argv){
    caffe2::GlobalInit(&argc,&argv);
    caffe2::caffe2_pretrained_run();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
