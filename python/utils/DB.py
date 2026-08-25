import numpy as np
import os
import sqlite3
import zlib

SIGNATURE_BYTES = 2
SIGNATURE = b'\x2B\x2B'
COLUMN_COUNT_BYTES = 1
TABLE_NAME_BYTES = 16
COMPRESS_BYTES = 1
BLOCK_NUM_BYTES = 4
OFFSET_BYTES = 8
ENCODING = 'ascii'


class DB:

    _query_data = None
    _query_fd = None
    _query_idx = 0

    _stream_insert_params = []
    _stream_insert_binary = []
    _stream_insert_offsets = []
    _stream_insert_size = 0

    types = {'int': 0x1, 'text': 0x2}

    def create_file(self, table_name: str, columns: list[list[str]], file_name: str = 'database', base_path: str = './', compress = False, compress_size = 1024 * 1024):
        self.bin_file = os.path.join(base_path, file_name) + '.bin'
        self.db_file = os.path.join(base_path, file_name) + '.db'
        self.config_file = os.path.join(base_path, 'config') + '.bin'
        self.query_data = None

        self.block_num = 0
        self.offset = 0

        assert(not os.path.exists(self.bin_file))
        assert(not os.path.exists(self.config_file))

        self.table_name = table_name
        tn = self.table_name.encode(ENCODING)
        assert(len(tn) <= TABLE_NAME_BYTES)
        tn += bytearray(TABLE_NAME_BYTES - len(tn))

        assert(len(columns) < (1 << (8 * COLUMN_COUNT_BYTES)))

        self.column_types = bytearray([self.types[type] for _, type in columns])
        self.compress = compress
        self.compress_size = compress_size
        
        # create config-file
        with open(self.config_file, 'wb') as fd:
            fd.write(SIGNATURE)
            fd.write(self.offset.to_bytes(OFFSET_BYTES))
            fd.write(self.block_num.to_bytes(BLOCK_NUM_BYTES, byteorder='big', signed=False))
            fd.write(tn)
            fd.write(len(columns).to_bytes(COLUMN_COUNT_BYTES, byteorder='big', signed=False))
            fd.write(self.column_types)
            fd.write(int.to_bytes(1 if self.compress else 0, length=1, byteorder='big', signed=False))
            fd.write(compress_size.to_bytes(4, byteorder='big', signed=False))
        
        # create bin-file
        open(self.bin_file, 'a').close()
            
        # create db-file
        self.con = sqlite3.connect(self.db_file)
        self.__create_sql_tables(table_name, columns)

    
    def connect(self, file_name = 'database', path = './'):
        self.bin_file = os.path.join(path, file_name) + '.bin'
        self.db_file = os.path.join(path, file_name) + '.db'
        self.config_file = os.path.join(path, 'config') + '.bin'
        self.query_data = None

        self.con = sqlite3.connect(self.db_file)
    
        with open(self.config_file, 'rb') as fd:
            assert(fd.read(SIGNATURE_BYTES) == SIGNATURE)
            self.offset = int.from_bytes(fd.read(OFFSET_BYTES), byteorder='big', signed=False)
            self.block_num = int.from_bytes(fd.read(BLOCK_NUM_BYTES), byteorder='big', signed=False)
            self.table_name = fd.read(TABLE_NAME_BYTES).decode(ENCODING).split('\x00')[0]
            column_count = int.from_bytes(fd.read(COLUMN_COUNT_BYTES), byteorder='big', signed=False)
            self.column_types = bytearray(fd.read(column_count))
            if fd.read(1) == b'\x01':
                self.compress = True
                self.compress_size = int.from_bytes(fd.read(4), byteorder='big', signed=False)
            else:
                self.compress = False
                self.compress_size = 0

    
    def query_seq(self, size_func, max_size):

        if self.data is None:
            cur = self.con.cursor()
            self.data = cur.execute(f'select id, start, end from {self.table_name}').fetchall()
            cur.close()
            self.bin_fd = open(self.bin_file, 'rb')
            self.last = 0
        
        end = len(self.data)
        size_total = 0

        mid = (end - self.last) // 2 + self.last
        while True:

            num_entrys = mid - self.last
            size = size_func(num_entrys, self.data[mid][2] - self.data[self.last][1])

            if size_total + size <= max_size:
                size_total += size

                if mid + 1 == len(self.data):
                    text_bin = self.bin_fd.read(size)
                    self.bin_fd.close()
                    x = self.data[self.last:mid]
                    self.data = None
                    return text_bin, x

                size = self.data[mid + 1][2] - self.data[self.last][1]
                if size_total + size_func(num_entrys + 1, size) > max_size:
                    text_bin = self.bin_fd.read(self.data[mid][1] - self.data[self.last][1])
                    tmp, self.last = self.last, mid + 1
                    return text_bin, self.data[tmp:mid]

                mid = (end - mid) // 2 + mid

            else:
                mid = (mid - self.last) // 2 + self.last


    
    def query_compr(self, max_size, reset=False):

        assert(max_size >= self.compress_size)

        # full reset because data might have changed
        if DB._query_idx == 0 or reset == True:
            query = 'select block_id, start, end, num_entrys, size_bytes from blocks;'
            DB._query_data = self.con.execute(query).fetchall()
            DB._query_fd = open(self.bin_file, 'rb')

        text_data = b''
        blocks, size_bytes = [], 0

        while True:

            if DB._query_idx == len(DB._query_data):
                DB._query_idx = 0
                DB._query_fd.close()
                break
            row = DB._query_data[DB._query_idx]
            block_num, start, end, num_entrys, size = row

            if size_bytes + size <= max_size:
                text_data += zlib.decompress(DB._query_fd.read(end - start))
                size_bytes += size
                blocks.append(block_num)
                DB._query_idx += 1
            else:
                break
        
        if len(blocks) == 0:
            return None

        return self.__read_entrys(blocks), text_data, size_bytes
        

        
    def __read_entrys(self, blocks):
        last_offset = 0
        data = []

        for block in blocks:
            query = f'select start + ?, end + ?, id, title, section_title from {self.table_name} where block_id = ?'
            data += self.con.execute(query, [last_offset, last_offset, block]).fetchall()
            last_offset = data[-1][1]
        return data
    
    def read_text_ids(self):
        lines = self.con.execute(f'select id from {self.table_name};').fetchall()
        return [line[0] for line in lines]

            
    
    def __create_sql_tables(self, table_name, columns):
        cols = ','.join([f'{column} {type}' for column, type in columns])
        if self.compress:
            stmts = [
                """
                create table blocks (
                    block_id int primary key,
                    start int,
                    end int,
                    num_entrys int,
                    size_bytes int
                ); """,
                f"""
                create table {table_name} (
                    block_id int,
                    start int,
                    end int,
                    {cols},
                    foreign key(block_id) references blocks(block_id)
                ); """,
                f"create index fd_blocks on {table_name}(block_id);"
            ]
        else:
            stmts = [
                f"""
                create table {table_name}(
                    {cols},
                    start int,
                    end int primary key
                ); """
            ]
        for stmt in stmts:
            self.con.execute(stmt)
        self.con.commit()

    
    def __update_config(self, new_offset, new_block_num):
        self.offset = new_offset
        self.block_num = new_block_num
        with open(self.config_file, 'ab') as fd:
            fd.seek(SIGNATURE_BYTES, 0)
            fd.write(self.offset.to_bytes(length=8, byteorder='big', signed=False))
            fd.seek(SIGNATURE_BYTES + OFFSET_BYTES, 0)
            fd.write(self.block_num.to_bytes(length=8, byteorder='big', signed=False))


    def __build_param_statement(self, num_params):
        return '(' + ','.join('?' for _ in range(num_params)) + '),'




    def stream_insert_compr(self, param, bin_content, size_func, batch_size = 1000):

        size = DB._stream_insert_size + size_func(1, len(bin_content))

        if size > self.compress_size:
            params = DB._stream_insert_params
            blocks = DB._stream_insert_binary
            offsets = DB._stream_insert_offsets
            self.__stream_insert_batch(params, b''.join(blocks), offsets, DB._stream_insert_size, batch_size)

            DB._stream_insert_params = [param]
            DB._stream_insert_binary = [bin_content]
            DB._stream_insert_offsets = [len(bin_content)]
            DB._stream_insert_size = size_func(1, len(bin_content))
        else:
            DB._stream_insert_params.append(param)
            DB._stream_insert_binary.append(bin_content)
            last = 0 if len(DB._stream_insert_offsets) == 0 else DB._stream_insert_offsets[-1]
            DB._stream_insert_offsets.append(last + len(bin_content))
            DB._stream_insert_size = size
    


    def stream_insert_finish(self, batch_size = 1000):
        if len(DB._stream_insert_params) > 0:
            params = DB._stream_insert_params
            blocks = DB._stream_insert_binary
            offsets = DB._stream_insert_offsets
            self.__stream_insert_batch(params, b''.join(blocks), offsets, DB._stream_insert_size, batch_size)
            DB._stream_insert_size = 0
            DB._stream_insert_binary = []
            DB._stream_insert_params = []
            DB._stream_insert_offsets = []

    
    def __stream_insert_batch(self, params, blocks, offsets, size, batch_size):

        compr = zlib.compress(blocks)

        with open(self.bin_file, 'ab') as fd:
            fd.write(compr)
            
        query = 'insert into blocks values ' + self.__build_param_statement(5)
        self.con.execute(query[:-1], [self.block_num, self.offset, self.offset + len(compr), len(params), size])


        query = f'insert into {self.table_name} values '
        q, values = query, []
        stmt = self.__build_param_statement(len(params[0]) + 3)

        end = 0

        for i in range(len(params)):
            start, end = end, offsets[i]
            
            q += stmt
            values += [self.block_num, start, end] + params[i]

            if (i + 1) % batch_size == 0 or i == len(params) - 1:
                self.con.execute(q[:-1], values)
                q, values = query, []
        
        self.con.commit()
        self.__update_config(self.offset + len(compr), self.block_num + 1)
    

    
    def __bulk_insert_compr(self, params: list[list], bin_content: list[bytes], batch_size=1000):

        if(len(params) == 0):
            return

        assert(len(params) == len(bin_content))

        cur = self.con.cursor()
        fd = open(self.bin_file, 'ab')
        offsets, blocks = np.zeros(len(bin_content)), np.zeros(len(bin_content))

        query = 'insert into blocks values '
        q = query
        stmt = self.__build_param_statement(5)
        values = []

        i, left = 0, 0
        while left < len(bin_content):
            right = left
            size_bytes = 0
            while right < len(bin_content) and size_bytes + len(bin_content[right]) <= self.compress_size:
                length = len(bin_content[right])
                offsets[right] = length
                size_bytes += length
                right += 1
            
            assert(left != right)

            compr = zlib.compress(b''.join(bin_content[left:right]))
            fd.write(compr)

            blocks[left:right].fill(self.block_num)
            q += stmt
            values += [self.block_num, self.offset, self.offset + len(compr), right - left, size_bytes]
            #q += self.__build_statement(values)

            i += 1
            if i >= batch_size or right == len(bin_content):
                i = 0
                cur.execute(q[:-1], values)
                q, values = query, []

            self.offset += len(compr)
            self.block_num += 1
            left = right

        fd.close()
        self.con.commit()
        self.__update_config(new_offset=self.offset, new_block_num=self.block_num)

        self.__data_insert_compr(params, blocks, offsets, batch_size)


    
    def __data_insert_compr(self, params, blocks, offsets, batch_size):

        cur = self.con.cursor()
        query = f'insert into {self.table_name} values '
        q, values = query, []
        i, last_block = 0, -1

        stmt = self.__build_param_statement(len(params[0]) + 3)
        for i in range(len(blocks)):
            start, end = (0, offsets[i]) if blocks[i] != last_block else (end, end + offsets[i])
            last_block = max(blocks[i], last_block)
            
            #q += self.__build_statement([blocks[i], start, end] + params[i])
            q += stmt
            values += [blocks[i], start, end] + params[i]

            if (i + 1) % batch_size == 0 or i == len(params) - 1:
                cur.execute(q[:-1], values)
                q, values = query, []

        cur.close()
        self.con.commit()

        

    def bulk_insert(self, params: list[list], bin_content: list[bytes], batch_size: int = 1000):
        if self.compress:
            self.__bulk_insert_compr(params, bin_content, batch_size)
        else:
            self.__bulk_insert_seq(params, bin_content, batch_size)

    def __bulk_insert_seq(self, params: list[list], bin_content: list[bytes], batch_size=1000):
        assert(len(params) == len(bin_content))
        offsets = [0] * len(bin_content)
        with open(self.bin_file, 'ab') as fd:
            for i, bin in enumerate(bin_content):
                offsets[i] = (self.offset if i == 0 else offsets[i - 1]) + len(bin)
                fd.write(bin)
        query = f'insert into {self.table_name} values '
        q = query
        cur = self.con.cursor()
        for i, param in enumerate(params): 
            q += '('
            for col in param:
                q += str(col).replace('\'', '') + ','
            # start
            q += str(offsets[i - 1] if i > 0 else self.offset) + ','
            # end
            q += str(offsets[i]) + '),'
            if (i + 1) % batch_size == 0 or i == len(params) - 1:
                cur.execute(q[:-1])
                q = query
        cur.close()
        self.con.commit()
        self.__update_config(new_offset=offsets[-1] + self.offset, new_block_num=self.block_num)

    def delete(self, remove_db = False):
        os.remove(self.bin_file)
        os.remove(self.config_file)
        if remove_db:
            os.remove(self.db_file)
        else:
            self.con.execute(f'drop table {self.table_name};')
            if self.compress:
                self.con.execute(f'drop table blocks;')
            self.con.commit()



