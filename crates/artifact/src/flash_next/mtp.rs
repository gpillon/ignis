//! The MTP head's companion container (spec flash-next/07, layout.md §13):
//! its pair pin against the main container, and its bind.
//!
//! A load with the head opens the companion beside the main container and,
//! before reading any byte of the head, holds the companion's sidecar
//! `pair.main` to the opened main container: `model_id`, `weights_id` and
//! `content_hash` ([`Reader::content_hash`], lowercase hex), each mismatch
//! refused by name. Then every object of the head is bound with its exact
//! format and shape -- the 29 tensors of [`mtp_entries`] and the 1024 expert
//! projections -- all on the device: the head's experts are resident for the
//! life of the load, outside the trunk's expert cache.

use std::collections::BTreeMap;

use super::{fail, mtp_entries, mtp_expert_name, FlashNextGeometry, Projection, ShapeRule, TrellisK, MTP_MODEL_ID};
use crate::binder::{Binder, MaterializationPlan, ObjectHandle};
use crate::{Reader, Result, StorageLayout};

/// One expert projection of the head: where it is, and its K class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpExpert {
    pub handle: ObjectHandle,
    pub k: TrellisK,
}

/// What [`bind`] placed: every object on the device, the 29 tensors by name,
/// and the experts in slot order (`expert * 2 + projection`, gate/up first).
pub struct MtpPlan {
    pub plan: MaterializationPlan,
    pub handles: BTreeMap<String, ObjectHandle>,
    pub experts: Vec<MtpExpert>,
}

/// The companion's sidecar `pair.main` against the main container `main`:
/// `Ok` when its `model_id`, `weights_id` and `content_hash` are the main
/// container's, else the first field that differs, both values named.
pub fn check_pair(sidecar: &serde_json::Value, main: &Reader) -> Result<()> {
    let pair = sidecar
        .pointer("/pair/main")
        .ok_or_else(|| fail("the MTP companion's sidecar records no pair.main"))?;
    let hash: String = main.content_hash().iter().map(|b| format!("{b:02x}")).collect();
    let identity = main.identity();
    for (field, actual) in [
        ("model_id", identity.model_id.as_str()),
        ("weights_id", identity.weights_id.as_str()),
        ("content_hash", hash.as_str()),
    ] {
        let recorded = pair
            .get(field)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| fail(format!("the MTP companion's pair.main records no {field}")))?;
        if recorded != actual {
            return Err(fail(format!(
                "the MTP companion is pinned to another main container: pair.main.{field} is {recorded}, the \
                 main container's is {actual}"
            )));
        }
    }
    Ok(())
}

/// Bind the companion: its identity, then every head object required with
/// its exact format, layout and shape, all placed on the device, and any
/// object the inventory does not name refused (ADR 0002).
pub fn bind(reader: &Reader, geometry: &FlashNextGeometry) -> Result<MtpPlan> {
    if reader.identity().model_id != MTP_MODEL_ID {
        return Err(fail(format!(
            "artifact model_id {} is not an MTP companion ({MTP_MODEL_ID})",
            reader.identity().model_id
        )));
    }
    let mut binder = Binder::new(reader);
    let mut handles = BTreeMap::new();
    for entry in mtp_entries(geometry) {
        let ShapeRule::Exact(shape) = &entry.shape else {
            return Err(fail(format!("{}: the head's tensors have exact shapes", entry.name)));
        };
        let handle = binder.require_tensor(&entry.name, entry.format, entry.layout, shape)?;
        binder.materialize_on_device(handle)?;
        handles.insert(entry.name.clone(), handle);
    }
    let mut experts = Vec::with_capacity(geometry.experts as usize * 2);
    for expert in 0..geometry.experts {
        for projection in Projection::ALL {
            let (handle, format) = binder.require_tensor_of(
                &mtp_expert_name(expert, projection),
                &TrellisK::ALL.map(TrellisK::format),
                StorageLayout::TrellisTile16V1,
                &geometry.projection_shape(projection),
            )?;
            binder.materialize_on_device(handle)?;
            experts.push(MtpExpert { handle, k: TrellisK::from_format(format).expect("a trellis format") });
        }
    }
    Ok(MtpPlan { plan: binder.finish()?, handles, experts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packer::{pack, sidecar_path, PackOutcome};

    /// A fixture companion packed beside its main container: the pin holds
    /// against that container and names the field that differs against any
    /// other; the bind places all 29 tensors and every expert projection,
    /// and refuses the main container as a companion.
    #[test]
    fn a_companion_binds_every_head_object_and_pins_its_main_container() {
        let main = super::super::fixture::build("mtp-bind").expect("the main fixture");
        main.tree.write_mtp(&main.path).expect("the head's work tree");
        let outcome = pack(&main.tree.mtp_pack_options(main.path.clone()), &mut |_| {}).expect("pack the head");
        assert!(matches!(outcome, PackOutcome::Finished { .. }), "{outcome:?}");
        let companion_path = main.tree.mtp_artifact_path();
        let companion = Reader::open(&companion_path).expect("open the companion");
        let main_reader = Reader::open(&main.path).expect("open the main container");
        let sidecar: serde_json::Value =
            serde_json::from_slice(&std::fs::read(sidecar_path(&companion_path)).expect("the sidecar")).expect("json");

        check_pair(&sidecar, &main_reader).expect("the companion is pinned to its own main container");
        for (field, value) in [("model_id", "another"), ("weights_id", "another"), ("content_hash", "00")] {
            let mut other = sidecar.clone();
            other["pair"]["main"][field] = serde_json::json!(value);
            let err = check_pair(&other, &main_reader).expect_err("another main container").to_string();
            assert!(err.contains(&format!("pair.main.{field} is {value}")), "{err}");
        }
        let err = check_pair(&serde_json::json!({}), &main_reader).expect_err("no pair").to_string();
        assert!(err.contains("no pair.main"), "{err}");

        let g = &main.tree.geometry;
        let plan = bind(&companion, g).expect("bind the companion");
        assert_eq!(plan.handles.len(), mtp_entries(g).len());
        assert_eq!(plan.experts.len(), g.experts as usize * 2);
        assert_eq!(plan.plan.device_objects.len(), plan.handles.len() + plan.experts.len(), "everything on the device");
        let err = bind(&main_reader, g).err().expect("the main container is no companion").to_string();
        assert!(err.contains("not an MTP companion"), "{err}");
    }
}
