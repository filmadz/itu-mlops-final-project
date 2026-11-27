// Dagger pipeline for the vial classifier: tests, training and image build.
package main

import (
	"context"

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
	// +ignore=[".git", "data/images/raw", "data/test/*.jpg", "artifacts", "models", "mlflow-state", "**/__pycache__", ".venv*"]
	source *dagger.Directory,
) (string, error) {
	return m.base(source).
		WithExec([]string{"python", "-m", "pytest", "-v"}).
		Stdout(ctx)
}
