// Dagger pipeline for the vial classifier: tests, training and image build.
package main

import (
	"context"
	"strconv"

	"dagger/vial-pipeline/internal/dagger"
)

type VialPipeline struct{}

// Python container with the project installed (CPU-only PyTorch)
func (m *VialPipeline) base(source *dagger.Directory) *dagger.Container {
	return dag.Container().
		From("python:3.13-slim").
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip-cache")).
		WithDirectory("/src", source).
		WithWorkdir("/src").
		WithExec([]string{
			"pip", "install", "-e", ".[dev]",
			"--extra-index-url", "https://download.pytorch.org/whl/cpu",
		})
}

// Runs the unit tests with pytest
func (m *VialPipeline) Test(
	ctx context.Context,
	// +defaultPath="/"
	// +ignore=[".git", "data/images/raw", "data/images/metadata.csv", "data/test/metadata.csv", "data/test/*.jpg", "artifacts", "models", "mlflow-state", "**/__pycache__", ".venv*"]
	source *dagger.Directory,
) (string, error) {
	return m.base(source).
		WithExec([]string{"python", "-m", "pytest", "-v"}).
		Stdout(ctx)
}

// Trains the model, promotes the best run and returns the exported model folder
func (m *VialPipeline) Train(
	// +defaultPath="/"
	// +ignore=[".git", "data/images/raw", "data/images/metadata.csv", "data/test/metadata.csv", "data/test/*.jpg", "artifacts", "models", "mlflow-state", "**/__pycache__", ".venv*"]
	source *dagger.Directory,
	// Number of training epochs
	// +default=5
	epochs int,
) *dagger.Directory {
	return m.base(source).
		WithEnvVariable("VIAL_NUM_EPOCHS", strconv.Itoa(epochs)).
		// .git is excluded from the container, so tell DVC not to look for it
		WithExec([]string{"dvc", "config", "--local", "core.no_scm", "true"}).
		WithExec([]string{"python", "-m", "vial_classifier.train"}).
		WithExec([]string{"python", "-m", "vial_classifier.registry", "promote"}).
		WithExec([]string{"python", "-m", "vial_classifier.registry", "export"}).
		Directory("/src/models/model")
}
