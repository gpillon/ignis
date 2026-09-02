//! Per-model semantic binder (ADR 0002).
//!
//! Rust port of the reference `binder` (`ninfer` `binder.h`/`binder.cpp`):
//! owns the per-model semantic layer the generic reader deliberately leaves
//! out — object-consumption validation, placement decisions, and the
//! materialization plan.
//!
//! Contract (ADR 0002): the binder must consume *every* object of the
//! reader's directory — [`Binder::finish`] fails if any object is left
//! unconsumed or has no placement. An unconsumed object is a load failure.

use crate::{
    align_up, fail, checked_add, NumericFormat, Object, PayloadSpan, Reader, Result,
    ResourceEncoding, StorageLayout,
};

/// A stable per-object handle (its directory index in `Reader::objects()`
/// order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectHandle {
    /// Directory index of the object (its position in `Reader::objects()`).
    pub index: usize,
}

/// One device-object placement in the plan's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePlacement {
    /// The placed object.
    pub handle: ObjectHandle,
    /// Offset inside the device arena (alignment-aware, binder-computed).
    pub offset: u64,
    /// Exact stored payload length.
    pub bytes: u64,
    /// File alignment applied to this object's offset (256 B for tensors).
    pub alignment: u64,
}

/// One host-retained resource placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPlacement {
    /// The retained resource.
    pub handle: ObjectHandle,
}

/// The binder's terminal output: placement decisions + device capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationPlan {
    /// Total directory objects (device + host placements).
    pub object_count: usize,
    /// Device arena size: the (offset + bytes) of the last device object,
    /// alignment-aware.
    pub device_capacity_bytes: u64,
    /// Device placements in binding order.
    pub device_objects: Vec<DevicePlacement>,
    /// Host-retained resources in binding order.
    pub host_objects: Vec<HostPlacement>,
}

/// The per-model binder: consumes every directory object and produces the
/// [`MaterializationPlan`] (ADR 0002).
pub struct Binder<'a> {
    reader: &'a Reader,
    consumed: Vec<bool>,
    planned: Vec<bool>,
    device_objects: Vec<DevicePlacement>,
    host_objects: Vec<HostPlacement>,
    capacity: u64,
}

impl<'a> Binder<'a> {
    /// Bind against an opened reader. Every object must be consumed before
    /// [`Binder::finish`] succeeds.
    pub fn new(reader: &'a Reader) -> Self {
        let object_count = reader.objects().len();
        Self {
            reader,
            consumed: vec![false; object_count],
            planned: vec![false; object_count],
            device_objects: Vec::new(),
            host_objects: Vec::new(),
            capacity: 0,
        }
    }

    /// Consume a directory object by exact name, validating it against the
    /// stored descriptor (the required format + layout + shape must match
    /// the stored tensor descriptor exactly).
    pub fn require_tensor(
        &mut self,
        name: &str,
        format: NumericFormat,
        layout: StorageLayout,
        shape: &[u64],
    ) -> Result<ObjectHandle> {
        let handle = self.consume(name)?;
        match &self.reader.objects()[handle.index] {
            Object::Tensor(t)
                if t.format == format
                    && t.layout == layout
                    && t.shape.as_slice() == shape =>
            {
                Ok(handle)
            }
            Object::Tensor(_) => Err(fail(format!(
                "tensor descriptor does not match target contract: {name}"
            ))),
            Object::Resource(_) => Err(fail(format!(
                "required tensor is a resource: {name}"
            ))),
        }
    }

    /// Consume a required resource by exact name, validating the encoding.
    pub fn require_resource(
        &mut self,
        name: &str,
        encoding: ResourceEncoding,
    ) -> Result<ObjectHandle> {
        let handle = self.consume(name)?;
        match &self.reader.objects()[handle.index] {
            Object::Resource(r) if r.encoding == encoding => Ok(handle),
            Object::Resource(_) => Err(fail(format!(
                "resource encoding does not match target contract: {name}"
            ))),
            Object::Tensor(_) => Err(fail(format!(
                "required resource is a tensor: {name}"
            ))),
        }
    }

    /// The directory descriptor of a consumed object.
    pub fn descriptor(&self, handle: ObjectHandle) -> &Object {
        &self.reader.objects()[handle.index]
    }

    /// The validated payload span of a consumed object (mapping-backed).
    pub fn payload(&self, handle: ObjectHandle) -> Result<PayloadSpan<'_>> {
        self.reader.payload_at(self.descriptor(handle))
    }

    /// Place a consumed tensor into the device arena (256-byte aligned).
    pub fn materialize_on_device(&mut self, handle: ObjectHandle) -> Result<()> {
        if self.planned[handle.index] {
            return Err(fail(format!(
                "artifact object has more than one materialization placement: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        let tensor = match &self.reader.objects()[handle.index] {
            Object::Tensor(t) => t,
            Object::Resource(_) => {
                return Err(fail(format!(
                    "resource cannot be materialized as a device tensor: {}",
                    self.reader.objects()[handle.index].name()
                )))
            }
        };
        let alignment = crate::tensor_alignment(tensor.layout);
        let offset = align_up(self.capacity, alignment, "materialization plan offset")?;
        let capacity = checked_add(offset, tensor.bytes, "materialization plan size")?;
        self.device_objects.push(DevicePlacement {
            handle,
            offset,
            bytes: tensor.bytes,
            alignment,
        });
        self.capacity = capacity;
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Retain a consumed resource on the host (its bytes stay in RAM; the
    /// reference `take_resource_bytes` pattern).
    pub fn retain_on_host(&mut self, handle: ObjectHandle) -> Result<()> {
        if self.planned[handle.index] {
            return Err(fail(format!(
                "artifact object has more than one materialization placement: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        if !matches!(&self.reader.objects()[handle.index], Object::Resource(_)) {
            return Err(fail(format!(
                "tensor cannot be retained as a host resource: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        self.host_objects.push(HostPlacement { handle });
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Mark a consumed object as validated-only (no placement, no upload).
    pub fn validate_only(&mut self, handle: ObjectHandle) -> Result<()> {
        if self.planned[handle.index] {
            return Err(fail(format!(
                "artifact object has more than one materialization placement: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Finish binding: every object consumed, every object placed.
    ///
    /// ADR 0002 — an unconsumed object is a load failure: `finish()` errors
    /// if any directory object was not consumed by an explicit
    /// `require_tensor` / `require_resource` call, or has no placement.
    pub fn finish(&self) -> Result<MaterializationPlan> {
        if let Some(i) = self.consumed.iter().position(|c| !c) {
            return Err(fail(format!(
                "artifact object was not consumed by the selected target: {}",
                self.reader.objects()[i].name()
            )));
        }
        if let Some(i) = self.planned.iter().position(|p| !p) {
            return Err(fail(format!(
                "artifact object has no materialization placement: {}",
                self.reader.objects()[i].name()
            )));
        }
        Ok(MaterializationPlan {
            object_count: self.consumed.len(),
            device_capacity_bytes: self.capacity,
            device_objects: self.device_objects.clone(),
            host_objects: self.host_objects.clone(),
        })
    }

    /// Consume a directory object by exact name (once — a second bind of the
    /// same object is an error).
    fn consume(&mut self, name: &str) -> Result<ObjectHandle> {
        let index = self
            .reader
            .index_of(name)
            .ok_or_else(|| fail(format!("required artifact object is missing: {name}")))?;
        if self.consumed[index] {
            return Err(fail(format!(
                "artifact object was bound more than once: {name}"
            )));
        }
        self.consumed[index] = true;
        Ok(ObjectHandle { index })
    }
}