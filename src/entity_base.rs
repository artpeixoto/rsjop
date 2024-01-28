use std::{array::TryFromSliceError, error::Error, fmt::Display, io::Read, iter, mem::size_of };
use ascii::{AsAsciiStr, IntoAsciiString};
use itertools::Itertools;
use ndarray::prelude::*;
use nom::{bytes::streaming::{tag, take}, error, number::complete::{le_f32, le_i32}, Err, IResult, Parser};
use num_traits::identities::{self, Zero};


#[derive(PartialEq, Clone, Debug)]
pub struct NpArrayD{
	pub unique_name: String,
	pub array_data:  NpArrayData,
	pub time_id: 	 i32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NpArrayDataType{
	U8, F32
}

impl NpArrayDataType{
	pub const fn data_size(&self) -> usize{
		match self{
			&NpArrayDataType::U8 => size_of::<u8>(),
			&NpArrayDataType::F32 => size_of::<f32>(),
		}
	}
}

#[derive(Clone, PartialEq, Debug)]
pub enum NpArrayData{
	U8(ArrayD<u8>),
	F32(ArrayD<f32>)
}

impl NpArrayData{
	pub fn len(&self) -> usize{
		match &self{
			NpArrayData::U8(arr) => arr.len(),
			NpArrayData::F32(arr) => arr.len(),
		}
	}
	pub const fn data_type(&self) -> NpArrayDataType{
		match self{
			NpArrayData::U8(_) => NpArrayDataType::U8,
			NpArrayData::F32(_) => NpArrayDataType::F32,
		}
	}
	pub fn shape(&self) -> &[usize]{
		match self{
			NpArrayData::U8(arr) => arr.shape(),
			NpArrayData::F32(arr) => arr.shape(),
		}
	}
	pub fn shape_is_1(&self) -> bool{
		self.shape() == &[1,1,1]
	}
}

impl NpArrayD{
	const _MAGIC_BYTES: &'static [u8] = b"Ihg\x1c"; 
	const _NAME_LEN: usize  = 128;	
	const _PRE_HEADER: usize = 17;	
	const _HEADER_END:  usize=  Self::_NAME_LEN + Self::_PRE_HEADER;

	pub fn new(name: String, arr: NpArrayData ) -> Self{
		Self { 
			unique_name: name,
			array_data: arr,
			time_id: 0,
		}
	}

	pub fn try_from_msg(msg: &[u8]) -> IResult<&[u8],Self> 
	{
		let msg = 
			tag::<&[u8],&[u8], ()>(Self::_MAGIC_BYTES)(msg)
			.map(|(msg, _)| msg)
			.unwrap_or(msg); 


		let (msg, msg_len) =	le_i32.map(|i32_val| i32_val as usize).parse(msg)?;
		let (msg, w ) = 		le_i32.map(|i32_val| i32_val as usize).parse(msg)?;
		let (msg, h) = 			le_i32.map(|i32_val| i32_val as usize).parse(msg)?;
		let (msg, c) = 			le_i32.map(|i32_val| i32_val as usize).parse(msg)?;
		let (msg, &[dt_byte,..]) = take(1_usize)(msg)? else {panic!()};

		let dt = match dt_byte {
			0 => NpArrayDataType::U8,
			_ => NpArrayDataType::F32,
		};

		let (msg, name) = 
			take(Self::_NAME_LEN)
			.map(|taken: &[u8]| 
				taken
				.into_ascii_string()
				.unwrap()
				.trim_end()
				.to_string()
			)
			.parse(msg)?;

		let (msg, arr) = { 
			if msg_len > 0 {
				fn make_array<T>(data: Vec<T>, dims: &[usize]) -> ArrayD<T> {
					Array::from_vec(data)
					.into_dyn()
					.into_shape(IxDyn(dims))
					.unwrap()
				}

				let (msg, arr_bytes) = take(msg_len).parse(msg)?;
				let dims = &[h, w, c];

				let arr = match dt {
					NpArrayDataType::F32 => {
						let arr_data = nom::combinator::iterator(arr_bytes, le_f32::<&[u8], ()>).collect_vec();
						NpArrayData::F32(make_array(arr_data, dims))
					}
					NpArrayDataType::U8 => { 
						let arr_data = 
							arr_bytes
							.to_owned();

						NpArrayData::U8(make_array(arr_data, dims))
					}
				};
				(msg, arr)
			} else {
				fn make_array<T>() -> ArrayD<T> where T: Clone + Zero {
					ArrayBase::zeros(IxDyn(&[0,0,0]))	
				}	
				let arr = match dt{
					NpArrayDataType::U8  => NpArrayData::U8(make_array()),
					NpArrayDataType::F32 => NpArrayData::F32(make_array()),
				};
				(msg, arr)
			}
		};
		
		let res = NpArrayD::new(name, arr);
		Ok((msg, res))
	}
	pub fn pack_msg(&self) -> Vec<u8>{
		let array_raw_data_len = {
			if self.array_data.shape_is_1() {
				0_usize
			} else{
				self.array_data.len() * self.array_data.data_type().data_size() 
			}
		};

		let (w, h, c) = {
			let mut data_dims = self.array_data.shape().iter();
			let h = data_dims.next().unwrap_or(&1);
			let w = data_dims.next().unwrap_or(&1);
			let c = data_dims.next().unwrap_or(&1);
			(w,h,c)
		};
	
		let dt_byte:u8 = 
			match self.array_data.data_type(){
				NpArrayDataType::U8 => 0,
				NpArrayDataType::F32 => 1,
			};

		fn bytes_from_usize ( value: &usize) -> impl Iterator<Item = u8>{
			(*value as i32).to_le_bytes().into_iter()
		}

		let name_bytes = { 
			let name_bytes = self.unique_name.as_ascii_str().unwrap().as_bytes().to_owned();
			let remaining_space_bytes =  
				core::iter::repeat(ascii::AsciiChar::Space.as_byte()).take(Self::_NAME_LEN - name_bytes.len());
			name_bytes.into_iter().chain(remaining_space_bytes)
		};
		let array_bytes: Box<dyn Iterator<Item=u8>> = {
			if self.array_data.shape_is_1() {
				Box::new(std::iter::empty::<u8>())
			} else {
				match &self.array_data{
					NpArrayData::U8(arr) => {Box::new(arr.iter().map(|b| *b))}
					NpArrayData::F32(arr) => {Box::new(arr.iter().map(|f32_val| f32_val.to_le_bytes()).flatten())}
				}
			}
		};

		let data = 
		 	iter::empty()
			.chain(Self::_MAGIC_BYTES.to_owned())
			.chain(bytes_from_usize(&array_raw_data_len))
			.chain(bytes_from_usize(&w))
			.chain(bytes_from_usize(&h))
			.chain(bytes_from_usize(&c) )
			.chain(std::iter::once(dt_byte))
			.chain(name_bytes)
			.chain(array_bytes);



		let mut res = 
			Vec::<u8>::with_capacity(
				Self::_HEADER_END + 
				Self::_MAGIC_BYTES.len() + 
				array_raw_data_len
			);
		res.extend(data);
	
		res
	}
} 

#[test]
fn test_msg(){
	let fucking_np_array = NpArrayD::new(
		"motherfucker".to_owned(),
		NpArrayData::F32(
			array![[[1.0, 2.0, 3.0]]].into_dyn()
		)	
	);
	let fucking_np_array_1 = NpArrayD::new(
		"motherfucker".to_owned(),
		NpArrayData::F32(
			array![[[1.0, 2.0, 4.0]]].into_dyn()
		)	
	);
	let fucking_np_array_msg = fucking_np_array.pack_msg();
	let fucking_np_array_msg_parse = NpArrayD::try_from_msg(&fucking_np_array_msg).unwrap().1;
	let fucking_np_array_msg_1 = fucking_np_array_1.pack_msg();
	let fucking_np_array_msg_parse_1 = NpArrayD::try_from_msg(&fucking_np_array_msg_1).unwrap().1;


	assert_eq!(fucking_np_array, fucking_np_array_msg_parse);
	assert_eq!(fucking_np_array_1, fucking_np_array_msg_parse_1);
	assert_ne!(fucking_np_array, fucking_np_array_msg_parse_1);
}

#[derive(core::fmt::Debug)]
pub struct JoyfulException{
	inner: Box<dyn Error + 'static>,
}
impl Display for JoyfulException{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		Ok(<Self as core::fmt::Debug>::fmt(&self, f).unwrap())
    }
}

impl Error for JoyfulException{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
		  Some(&*self.inner)
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn Error> {
        self.source()
    }
}

fn env_has_flags(flag: &str) -> bool {
	let args = std::env::args().collect_vec();
	args.len() > 1 && args.iter().any(|arg| flag == arg)
}

pub fn is_custom_level_runner() -> bool{
	env_has_flags(&"custom_level_running")
}

pub fn internal_rust_process() -> bool{
	env_has_flags(&"internal_rust")
}
pub mod entity{
	pub trait Entity{
	}
	pub struct EntityClass<T>{
		pub 
	}
	pub impl 
	pub struct EntityBaseData{
		entity_name: String,
		class_name : &'static str,
	}
	impl EntityBaseData{
		
	}
}
