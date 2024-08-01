import UIKit
import AVFoundation
import CoreML
import Vision
import Photos

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var requests = [VNRequest]()
    var overlayLabel: UILabel!
    var photoOutput = AVCapturePhotoOutput()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupVision()
        setupOverlayLabel()
        setupCaptureButton()
        requestPhotoPermissions()
        setupPinchToZoom()
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let backCamera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: backCamera),
              captureSession.canAddInput(input) else {
            fatalError("Failed to add camera input to session")
        }
        
        captureSession.addInput(input)
        captureSession.addOutput(photoOutput)

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        captureSession.startRunning()
    }
    
    func setupPinchToZoom() {
        let pinchRecognizer = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch))
        view.addGestureRecognizer(pinchRecognizer)
    }

    @objc func handlePinch(pinch: UIPinchGestureRecognizer) {
        guard let device = AVCaptureDevice.default(for: .video) else { return }

        if pinch.state == .changed {
            let maxZoomFactor = device.activeFormat.videoMaxZoomFactor
            let pinchVelocityDividerFactor: CGFloat = 5.0

            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }

                let desiredZoomFactor = device.videoZoomFactor + atan2(pinch.velocity, pinchVelocityDividerFactor)
                device.videoZoomFactor = max(1.0, min(desiredZoomFactor, maxZoomFactor))
            } catch {
                print("Failed to lock device for configuration: \(error)")
            }
        }
    }


    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get image buffer from sample buffer")
            return
        }
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try requestHandler.perform(self.requests)
        } catch {
            print("Failed to perform Classification: \(error)")
        }
    }

    func setupVision() {
        guard let visionModel = try? VNCoreMLModel(for: NumbersLettersAndSymbols_1().model) else {
            fatalError("Can't load ML model")
        }
        let classificationRequest = VNCoreMLRequest(model: visionModel, completionHandler: self.processClassifications)
        classificationRequest.imageCropAndScaleOption = .centerCrop
        self.requests.append(classificationRequest)
    }

    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            if let error = error {
                print("Error in processing request: \(error.localizedDescription)")
                self.overlayLabel.text = "Error: \(error.localizedDescription)"
                return
            }
            guard let results = request.results as? [VNClassificationObservation], !results.isEmpty else {
                print("No results")
                self.overlayLabel.text = "No recognizable objects."
                return
            }
            
            if let topResult = results.max(by: { a, b in a.confidence < b.confidence }) {
                let formattedString = String(format: "Detected: %@ (%.2f%%)", topResult.identifier, topResult.confidence * 100)
                print(formattedString)
                self.overlayLabel.text = formattedString
            } else {
                self.overlayLabel.text = "No recognizable objects."
            }
        }
    }

    func setupOverlayLabel() {
        overlayLabel = UILabel(frame: CGRect(x: 20, y: 80, width: view.frame.width - 40, height: 100))
        overlayLabel.textAlignment = .center
        overlayLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        overlayLabel.textColor = .white
        overlayLabel.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        overlayLabel.numberOfLines = 0
        view.addSubview(overlayLabel)
        view.bringSubviewToFront(overlayLabel)
    }

    func setupCaptureButton() {
        let captureButton = UIButton(frame: CGRect(x: (view.frame.width - 70) / 2, y: view.frame.height - 120, width: 70, height: 70))
        captureButton.layer.cornerRadius = 35
        captureButton.backgroundColor = .white
        captureButton.layer.borderWidth = 3
        captureButton.layer.borderColor = UIColor.black.cgColor
        captureButton.addTarget(self, action: #selector(capturePhoto), for: .touchUpInside)
        view.addSubview(captureButton)
        view.bringSubviewToFront(captureButton)
    }

    @objc func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        flashScreen()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    func flashScreen() {
        let flashView = UIView(frame: self.view.bounds)
        flashView.backgroundColor = .white
        flashView.alpha = 1
        view.addSubview(flashView)
        UIView.animate(withDuration: 0.75, animations: {
            flashView.alpha = 0
        }) { _ in
            flashView.removeFromSuperview()
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error capturing photo: \(error.localizedDescription)")
            return
        }
        
        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            print("Error capturing photo: imageData not found")
            return
        }

        saveImageWithOverlay(image: image)
    }
    
    func imageWithText(_ image: UIImage, text: String) -> UIImage? {
        // Setup the image context using the passed image.
        let scale = UIScreen.main.scale
        UIGraphicsBeginImageContextWithOptions(image.size, false, scale)

        // Draw the image in the current context as background
        image.draw(at: CGPoint.zero)

        // Setup the font attributes for the text drawing
        let textColor: UIColor = .white
        let textFont: UIFont = UIFont.boldSystemFont(ofSize: 40)

        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.alignment = .center

        let textAttributes = [
            NSAttributedString.Key.font: textFont,
            NSAttributedString.Key.foregroundColor: textColor,
            NSAttributedString.Key.paragraphStyle: paragraphStyle,
            NSAttributedString.Key.backgroundColor: UIColor.black.withAlphaComponent(0.5)
        ]

        // Create a point within the space that is as big as the image to draw the text
        let rect = CGRect(x: 0, y: (image.size.height / 2) - 20, width: image.size.width, height: 40)
        text.draw(in: rect, withAttributes: textAttributes)

        // Capture the image and end the context
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage
    }


    func saveImageWithOverlay(image: UIImage) {
        guard let finalImage = imageWithText(image, text: overlayLabel.text ?? "No Data") else {
            print("Error creating final image with overlay")
            return
        }
        UIImageWriteToSavedPhotosAlbum(finalImage, nil, nil, nil)
    }

    func requestPhotoPermissions() {
        PHPhotoLibrary.requestAuthorization { status in
            if status != .authorized {
                print("Cannot save photos. Authorization denied.")
            }
        }
    }
}
