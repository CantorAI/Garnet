#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace PrimaryBlocks {
constexpr size_t Words=512,RawBytes=1024,CompactBytes=768;
struct Block {std::array<uint8_t,RawBytes> data{};uint64_t metadata=0;size_t bytes=0;bool compact=false;};
inline bool Eligible(uint64_t rows,uint64_t inner){return rows&&inner&&rows%4==0&&inner%128==0;}
inline uint64_t Offset(uint64_t metadata){return (metadata>>9)*256;}
inline Block Encode(const std::array<uint16_t,Words>& words,uint64_t offset){
 if(offset%256||offset/256>(std::numeric_limits<uint64_t>::max()>>9))throw std::invalid_argument("Unrepresentable primary block offset");
 uint16_t low=255,high=0;
 for(auto bits:words){const auto e=uint16_t((bits>>7)&255);if(e==255)throw std::invalid_argument("Nonfinite BF16 block");if(e<low)low=e;if(e>high)high=e;}
 Block block;block.compact=(high-low<=15);block.bytes=block.compact?CompactBytes:RawBytes;
 block.metadata=((offset/256)<<9)|(block.compact?(uint64_t(low)|256):0);
 if(block.compact){
  for(size_t i=0;i<Words;++i){auto bits=words[i];block.data[i]=uint8_t((bits&127)|((bits>>8)&128));auto tag=uint8_t(((bits>>7)&255)-low);block.data[512+i/2]|=uint8_t(tag<<((i%2)*4));}
 }else for(size_t i=0;i<Words;++i){block.data[i*2]=uint8_t(words[i]);block.data[i*2+1]=uint8_t(words[i]>>8);}
 return block;
}
inline size_t Length(uint64_t metadata){if(metadata&256){if((metadata&255)>254)throw std::invalid_argument("Nonfinite compact base");return CompactBytes;}if(metadata&255)throw std::invalid_argument("Raw metadata has an exponent base");return RawBytes;}
inline uint16_t DecodeLocal(uint64_t metadata,const uint8_t* bytes,size_t length,size_t word){
 if(!bytes||word>=Words||length<Length(metadata))throw std::out_of_range("Primary block decode bounds");
 uint16_t bits;
 if(metadata&256){const uint8_t sm=bytes[word];const unsigned e=unsigned(metadata&255)+((bytes[512+word/2]>>((word%2)*4))&15);if(e>=255)throw std::invalid_argument("Nonfinite decoded exponent");bits=uint16_t((sm&127)|((sm&128)<<8)|(e<<7));}
 else bits=uint16_t(bytes[word*2])|uint16_t(bytes[word*2+1])<<8;
 if(((bits>>7)&255)==255)throw std::invalid_argument("Nonfinite raw block value");
 return bits;
}
inline uint16_t Decode(uint64_t metadata,const uint8_t* payload,size_t length,size_t word){
 const auto offset=Offset(metadata);const auto needed=Length(metadata);
 if(offset>length||needed>length-offset)throw std::out_of_range("Primary matrix payload bounds");
 return DecodeLocal(metadata,payload+offset,needed,word);
}
}
