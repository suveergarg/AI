from dataclasses import dataclass
import dataclasses
from io import BytesIO
from struct import *
import socket

#https://implement-dns.wizardzines.com/book/part_2

@dataclass
class DNSHeader:
    id: int= 22
    flags: int = 0 | (1<<8)
    num_ques: int = 1
    ans_record: int = 0
    aut_record: int = 0
    add_record: int = 0

@dataclass
class DNSQuestion:
    question: bytes
    query_type: int = 1
    query_class: int = 1

@dataclass
class DNSRecord:
    name: bytes
    type_: int
    class_: int
    ttl: int
    data: bytes 

def encode_dns_name(domain_name):
    encoded = b""
    for part in domain_name.encode("ascii").split(b"."):
        encoded += bytes([len(part)]) + part
    return encoded + b"\x00"

def parse_header(reader):
    items = unpack("!HHHHHH", reader.read(12))
    # see "a note on BytesIO" for an explanation of `reader` here
    return DNSHeader(*items)

def decode_name_simple(reader):
    parts = []
    while (length := reader.read(1)[0]) != 0:
        parts.append(reader.read(length))
    return b".".join(parts)

def decode_name(reader):
    parts = []
    while (length := reader.read(1)[0]) != 0:
        if length & 0b1100_0000:
            parts.append(decode_compressed_name(length, reader))
            break
        else:
            parts.append(reader.read(length))
    return b".".join(parts)

def decode_compressed_name(length, reader):
    pointer_bytes = bytes([length & 0b0011_1111]) + reader.read(1)
    pointer = unpack("!H", pointer_bytes)[0]
    current_pos = reader.tell()
    reader.seek(pointer)
    result = decode_name(reader)
    reader.seek(current_pos)
    return result

def parse_question(reader):
    name = decode_name_simple(reader)
    data = reader.read(4)
    type_, class_ = unpack("!HH", data)
    return DNSQuestion(name, type_, class_)

def parse_record(reader):
    name = decode_name(reader)
    # the the type, class, TTL, and data length together are 10 bytes (2 + 2 + 4 + 2 = 10)
    # so we read 10 bytes
    data = reader.read(10)
    # HHIH means 2-byte int, 2-byte-int, 4-byte int, 2-byte int
    type_, class_, ttl, data_len = unpack("!HHIH", data) 
    data = reader.read(data_len)
    return DNSRecord(name, type_, class_, ttl, data)

def build_query(domain_name):
    dns_header = DNSHeader()
    dns_question = DNSQuestion(question = encode_dns_name(domain_name))
    header_fields = dataclasses.astuple(dns_header)

    header_bytes = pack('!HHHHHH' , *header_fields)
    question_bytes = dns_question.question + pack("!HH", dns_question.query_type, dns_question.query_class)

    dns_bytes = header_bytes+question_bytes
    return dns_bytes

query = build_query('dns.google.com')
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(query, ("8.8.8.8", 53))
response, _ = sock.recvfrom(1024)
reader = BytesIO(response)
response_header = parse_header(reader)
response_question = parse_question(reader)
response_record = parse_record(reader)

print(response_header.id)
print(response_question)
print(response_record)