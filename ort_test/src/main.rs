use std::{
    error::Error,
    io::{self, ErrorKind},
};

#[cfg(all(target_arch = "riscv64", target_env = "musl"))]
use ort::AsPointer;
#[cfg(all(target_arch = "riscv64", target_env = "musl"))]
use std::ffi::CStr;

use image::{Rgb, RgbImage, imageops::FilterType};
use ndarray::{Array, Array4};
use ort::inputs;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::TensorRef,
};

#[cfg(all(target_arch = "riscv64", target_env = "musl"))]
unsafe extern "C" {
    fn OrtSessionOptionsAppendExecutionProvider_K3(
        options: *mut ort::sys::OrtSessionOptions,
    ) -> ort::sys::OrtStatusPtr;
}
#[cfg(all(target_arch = "riscv64", target_env = "musl"))]
unsafe extern "C" {
    fn k3_ort_executed_node_count() -> u64;
}

#[cfg(all(target_arch = "riscv64", target_env = "musl"))]
fn append_k3(builder: &mut SessionBuilder) -> Result<(), Box<dyn Error>> {
    let status = unsafe {
        // builder 持有有效的 OrtSessionOptions；注册发生在 commit 前。
        OrtSessionOptionsAppendExecutionProvider_K3(builder.ptr_mut())
    };

    if status.0.is_null() {
        return Ok(());
    }

    let message = unsafe {
        let message = CStr::from_ptr((ort::api().GetErrorMessage)(status.0))
            .to_string_lossy()
            .into_owned();
        (ort::api().ReleaseStatus)(status.0);
        message
    };
    Err(Box::new(io::Error::other(message)))
}

#[cfg(not(all(target_arch = "riscv64", target_env = "musl")))]
fn append_k3(_: &mut SessionBuilder) -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[cfg(all(target_arch = "riscv64", target_env = "musl"))]
fn executed_k3_node_count() -> u64 {
    unsafe { k3_ort_executed_node_count() }
}

#[cfg(not(all(target_arch = "riscv64", target_env = "musl")))]
fn executed_k3_node_count() -> u64 {
    0
}

fn parse_test_flag() -> Result<bool, io::Error> {
    let mut run_concat_test = false;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-t" => run_concat_test = true,
            "-h" | "--help" => {
                println!("Usage: ort_learn [-t]\n\n  -t  run concat.onnx instead of YOLOv5");
                std::process::exit(0);
            }
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown option: {arg} (use -h for help)"),
                ));
            }
        }
    }

    Ok(run_concat_test)
}

/// 解析 YOLOv5 `[1, 25200, 85]` 输出，做 class-aware NMS，并写出带框图片。
fn draw_yolo_boxes(image: &mut RgbImage, predictions: &[f32]) -> Result<(), Box<dyn Error>> {
    const ATTRIBUTES: usize = 85;
    const CONFIDENCE: f32 = 0.25;
    const NMS_IOU: f32 = 0.45;

    if predictions.len() % ATTRIBUTES != 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidData,
            "expected YOLOv5 predictions with 85 values per box",
        )));
    }

    let max_x = image.width().saturating_sub(1) as f32;
    let max_y = image.height().saturating_sub(1) as f32;
    let mut candidates = Vec::new();
    for row in predictions.chunks_exact(ATTRIBUTES) {
        let (mut class, mut class_score) = (0, 0.0_f32);
        for (index, &score) in row[5..].iter().enumerate() {
            if score.is_finite() && score > class_score {
                (class, class_score) = (index, score);
            }
        }
        let score = row[4] * class_score;
        if !score.is_finite()
            || score < CONFIDENCE
            || !row[..4].iter().all(|value| value.is_finite())
        {
            continue;
        }
        let x1 = (row[0] - row[2] / 2.0).clamp(0.0, max_x) as u32;
        let y1 = (row[1] - row[3] / 2.0).clamp(0.0, max_y) as u32;
        let x2 = (row[0] + row[2] / 2.0).clamp(0.0, max_x) as u32;
        let y2 = (row[1] + row[3] / 2.0).clamp(0.0, max_y) as u32;
        if x1 < x2 && y1 < y2 {
            candidates.push((x1, y1, x2, y2, class, score));
        }
    }

    candidates.sort_by(|left, right| right.5.total_cmp(&left.5));
    let mut boxes = Vec::new();
    'candidate: for candidate @ (x1, y1, x2, y2, class, _) in candidates {
        for &(kept_x1, kept_y1, kept_x2, kept_y2, kept_class, _) in &boxes {
            if class != kept_class {
                continue;
            }
            let intersection = x2.min(kept_x2).saturating_sub(x1.max(kept_x1)) as f32
                * y2.min(kept_y2).saturating_sub(y1.max(kept_y1)) as f32;
            let union = (x2 - x1) as f32 * (y2 - y1) as f32
                + (kept_x2 - kept_x1) as f32 * (kept_y2 - kept_y1) as f32
                - intersection;
            if union > 0.0 && intersection / union > NMS_IOU {
                continue 'candidate;
            }
        }
        boxes.push(candidate);
    }

    // 不引入额外绘图库：直接画红色矩形边框。
    for (x1, y1, x2, y2, _, _) in &boxes {
        for x in *x1..=*x2 {
            image.put_pixel(x, *y1, Rgb([255, 0, 0]));
            image.put_pixel(x, *y2, Rgb([255, 0, 0]));
        }
        for y in *y1..=*y2 {
            image.put_pixel(*x1, y, Rgb([255, 0, 0]));
            image.put_pixel(*x2, y, Rgb([255, 0, 0]));
        }
    }
    image.save("detected.png")?;
    println!("YOLOv5 detections: {}; wrote detected.png", boxes.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let run_concat_test = parse_test_flag()?;

    let mut builder = Session::builder()?.with_intra_threads(1)
    ?.with_profiling("profile.json")?;
    append_k3(&mut builder)?;

    if run_concat_test {
        let mut session = builder.commit_from_file("./concat.onnx")?;
        let a = Array::from_shape_vec((2, 1, 4, 4), (0..32).map(|x| x as f32).collect())?;
        let b = Array::from_shape_vec((2, 5, 4, 4), (0..160).map(|x| x as f32).collect())?;
        println!("concat model load success");

        let output = session.run(inputs![
            "A" => TensorRef::from_array_view(&a)?,
            "B" => TensorRef::from_array_view(&b)?,
        ])?;
        let tensor = output[0].try_extract_tensor::<f32>()?;
        println!("Concat output tensor: {tensor:?}");
        println!("{output:?}");
    } else {
        let image: RgbImage = image::open("bus.png")?.to_rgb8();
        let mut image = image::imageops::resize(&image, 640, 640, FilterType::Triangle);
        let input = Array4::from_shape_fn((1, 3, 640, 640), |(_, channel, y, x)| {
            f32::from(image.get_pixel(x as u32, y as u32)[channel]) / 255.0
        });
        println!("image load success");

        let mut session = builder.commit_from_file("./yolov5n.onnx")?;
        println!("model load success");

        let output = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let (shape, predictions) = output[0].try_extract_tensor::<f32>()?;
        println!("YOLOv5 output shape: {shape:?}");
        draw_yolo_boxes(&mut image, predictions)?;
    }

    println!("K3 executed nodes: {}", executed_k3_node_count());
    Ok(())
}
