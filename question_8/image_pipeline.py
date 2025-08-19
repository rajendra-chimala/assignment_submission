
from pipeline_class import ImagePipeline

if __name__ == "__main__":
    INPUT_DIR = "images"
    OUTPUT_DIR = "batch_results"

    print("Starting image analysis pipeline...")
    pipeline = ImagePipeline(INPUT_DIR, OUTPUT_DIR)

    summary, collage_images = pipeline.process_images()
    summary_np, eigenvalues, most_edges_filename = pipeline.analyze_features()
    report_path = pipeline.save_report(summary_np, eigenvalues, most_edges_filename)

    rows = len(summary)
    pipeline.create_collage(collage_images, rows=rows, cols=2, title="collage.jpg")

    print("Pipeline completed successfully.")
    print(f"Report saved to: {report_path}")
    print(f"Collage saved to: {OUTPUT_DIR}/collage.jpg")
