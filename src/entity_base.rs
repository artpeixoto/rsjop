use ndarray::prelude::*;
pub struct NpArrayD{
	pub unique_name: String,
	pub array_data:  ArrayD<u8>,
	pub time_id: 	 i32,
}


impl NpArrayD{
	const _MAGIC_BYTES: &'static [u8] = &[]
} 
