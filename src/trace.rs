use serde::{Deserialize, Serialize};
use slog::{Drain, Key, OwnedKVList, Record, SerdeValue, Serializer, KV};
use std::fmt;

#[derive(Clone, Debug, Deserialize, Serialize, slog_derive::SerdeValue)]
pub enum Event {
    WmeAdded,
    WmeRemoved,
    ProductionAdded,
    ProductionRemoved,
}

impl KV for Event {
    fn serialize(&self, _: &Record, serializer: &mut Serializer) -> slog::Result {
        serializer.emit_serde("event", self)
    }
}

pub struct ObserverDrain;

impl Drain for ObserverDrain {
    type Ok = ();
    type Err = slog::Error;

    fn log(&self, record: &Record, _: &OwnedKVList) -> Result<Self::Ok, Self::Err> {
        struct S;
        impl Serializer for S {
            fn emit_arguments(&mut self, _: Key, _: &fmt::Arguments) -> slog::Result {
                Ok(())
            }

            fn emit_serde(&mut self, _: Key, value: &SerdeValue) -> slog::Result {
                let buffer = serde_json::to_string(value.as_serde()).unwrap();
                let event: Event = serde_json::from_str(&buffer).map_err(|_| slog::Error::Other)?;
                println!("{:?}", event);
                Ok(())
            }
        }
        record.kv().serialize(record, &mut S)
    }
}
