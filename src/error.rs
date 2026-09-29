//! Error types for the fgrain simulation engine.
//!
//! Provides structured, type-safe errors utilizing [`thiserror`].

use thiserror::Error;

/// Errors arising during GPU hardware initialization or compute shader execution.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GpuError {
    /// Failed to find a compatible GPU adapter matching required specifications.
    #[error("Failed to find suitable GPU adapter: {0}")]
    AdapterNotFound(String),

    /// Failed to acquire a logical GPU device or command queue from the adapter.
    #[error("Failed to create wgpu device: {0}")]
    DeviceRequestFailed(String),

    /// Error during asynchronous GPU completion or channel synchronization.
    #[error("GPU channel error: {0}")]
    ChannelError(String),

    /// Error mapping GPU staging buffer memory to host CPU address space.
    #[error("GPU buffer map error: {0}")]
    BufferMapError(String),

    /// Error creating or accessing slice bounds on a GPU buffer.
    #[error("GPU buffer range error: {0}")]
    BufferRangeError(String),

    /// General compute pipeline or shader compilation failure.
    #[error("GPU pipeline error: {0}")]
    PipelineError(String),
}

/// Errors arising during image loading, rendering pipeline dispatch, or file saving.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Failure during image decoding, encoding, or format manipulation.
    #[error("Image I/O error: {0}")]
    Image(#[from] image::ImageError),

    /// Standard filesystem I/O error.
    #[error("Filesystem I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error occurred within the GPU compute backend.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// Invalid configuration parameters provided to the pipeline.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_formatting() {
        let err = GpuError::AdapterNotFound("No Vulkan/DX12 device".to_string());
        assert_eq!(
            err.to_string(),
            "Failed to find suitable GPU adapter: No Vulkan/DX12 device"
        );

        let pipe_err = PipelineError::from(err);
        assert!(pipe_err.to_string().starts_with("GPU error:"));
    }
}
