use rustacuda::prelude::*;
use std::ffi::CString;
#[macro_use]
extern crate rustacuda;
use image::ImageReader;
use clap::Parser;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// input file folder
    #[arg(short, long)]
    inputs: String,

    /// output file folder
    #[arg(short, long)]
    outputs: String,

    /// Number of streams
    #[arg(short, long, default_value_t = 1)]
    streams: u8,
}

// init cuda and load kernel
fn init() -> (Device, Context, Module) {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty()).expect("No CUDA error");
    // Get the first device
    let device = Device::get_device(0).expect("No CUDA device found");
    // Create a context associated to this device
    let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).expect("Failed to create CUDA context");
    // Load the module containing the kernel. This is a PTX file which contains the compiled kernel.
    let ptx = CString::new(include_str!("../kernels/filter.ptx")).expect("Failed to create CString from PTX string");
    let module = Module::load_from_string(&ptx).expect("Failed to load module");
    (device, context, module)
}

// create count number of streams to run parallel processing
fn create_streams(count:u32) -> Vec<Stream> {
    let mut result = Vec::new();
    for _ in 0..count {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream");
        result.push(stream);
    }
    return result;
}

// load image and return (width,height,bytes)
fn load_image(file_path: std::path::PathBuf) -> (u32, u32, Vec<u8>) {
    let img = ImageReader::open(&file_path).expect("Failed to open image").decode().expect("Failed to decode image");
    let (width, height) = (img.width(), img.height());
    return (width, height, img.as_bytes().to_vec());
}

// load and start async processing for an image and a certain stream
fn process_image_async(file_path: std::path::PathBuf, module: &Module, stream: &Stream) -> (std::path::PathBuf,u32,u32,DeviceBuffer<u8>) {
    // load image
    let (width, height, img_bytes) = load_image(file_path.clone());

    println!("Process: {} ({}x{})", file_path.display(),width,height);

    let mut d_out_image_ptr: DeviceBuffer<u8>;

    unsafe{
        let bytes = img_bytes.as_slice();
        let size = bytes.len();
        let mut d_source_image_ptr = DeviceBuffer::from_slice_async(bytes, stream).expect("alloc device memory error");
        d_out_image_ptr = DeviceBuffer::uninitialized(size).expect("alloc device memory error");

        launch!(module.grayscale<<<((width+31_u32)/32_u32,(height+31_u32)/32_u32,1), (32,32,1), 0, stream>>>(
            // Kernel arguments
            d_source_image_ptr.as_device_ptr(),
            d_out_image_ptr.as_device_ptr(),
            width,
            height
        )).expect("kernel launch error");
    };

    return (file_path, width, height, d_out_image_ptr);
}

// wait all streams finished
fn wait(streams:&Vec<Stream>) {
    for stream in streams {
        stream.synchronize().expect("Failed to synchronize stream");
    }   
}

fn save_image(output_folder: &String, file_path: std::path::PathBuf, width:u32, height:u32, image_ptr: DeviceBuffer<u8>) {
    let mut out_host = Vec::with_capacity(image_ptr.len());
    out_host.resize(image_ptr.len(), 0);
    image_ptr.copy_to(&mut out_host).expect("Failed to copy data from device to host");

    // Save the output image
    std::fs::create_dir_all(output_folder).expect("Failed to create output directory");
    let out_img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, out_host).expect("Failed to create output image");
    let out_path = format!("{}/{}", output_folder, file_path.file_name().unwrap().to_str().unwrap());
    out_img.save(&out_path).expect("Failed to save output image");
}

fn main() {
    let args = Args::parse();

    let (_device, _context, module) = init();

    let streams = create_streams(args.streams.into());

    let mut results = Vec::new();

    // iterate through input dir and initiate kernel process async
    let folder = std::fs::read_dir(args.inputs).unwrap();
    let mut current_stream_index = 0;
    for file in folder {

        // check path is file
        let file_path = file.unwrap().path();
        if file_path.is_file() {

            // load image and start cuda kernel
            let result = process_image_async(file_path, &module, &streams[current_stream_index]);

            // save result pointers for later
            results.push(result);

            // use next stream
            current_stream_index+=1;
            if current_stream_index >= streams.len() {
                current_stream_index = 0;
            }
        }
    }

    // wait all kernels finished
    wait(&streams);

    // save all results as image
    for result in results {

        let (file_path, width, height, image_ptr) = result;

        save_image(&args.outputs, file_path,width,height,image_ptr);
    }


}
