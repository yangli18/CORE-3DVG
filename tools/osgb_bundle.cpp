#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Matrix>
#include <osg/NodeVisitor>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

namespace {

constexpr std::size_t kChunkSize = 64 * 1024;
constexpr const char* kSchemaKey = "core3dvg.bundle.schema";
constexpr const char* kEncodingKey = "core3dvg.bundle.encoding";
constexpr const char* kChunkCountKey = "core3dvg.bundle.chunk_count";
constexpr const char* kPayloadSizeKey = "core3dvg.bundle.payload_size";
constexpr const char* kChunkPrefix = "core3dvg.bundle.payload.";

int inspect(const std::string& input_path);

struct PointRecord {
    float x;
    float y;
    float z;
    std::uint8_t red;
    std::uint8_t green;
    std::uint8_t blue;
    std::uint8_t alpha;
};

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open payload file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_file(const std::string& path, const std::string& data) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create payload file: " + path);
    }
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!output) {
        throw std::runtime_error("cannot write payload file: " + path);
    }
}

std::string chunk_key(std::size_t index) {
    std::ostringstream key;
    key << kChunkPrefix << std::setw(6) << std::setfill('0') << index;
    return key.str();
}

class GeometryStatsVisitor : public osg::NodeVisitor {
public:
    GeometryStatsVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN),
          geometry_count(0),
          vertex_count(0),
          primitive_set_count(0) {}

    void apply(osg::Geode& geode) override {
        for (unsigned int i = 0; i < geode.getNumDrawables(); ++i) {
            osg::Geometry* geometry = geode.getDrawable(i)->asGeometry();
            if (geometry == nullptr) {
                continue;
            }
            ++geometry_count;
            if (geometry->getVertexArray() != nullptr) {
                vertex_count += geometry->getVertexArray()->getNumElements();
            }
            primitive_set_count += geometry->getNumPrimitiveSets();
        }
        traverse(geode);
    }

    std::size_t geometry_count;
    std::size_t vertex_count;
    std::size_t primitive_set_count;
};

std::uint8_t float_color_to_byte(float value) {
    value = std::max(0.0f, std::min(1.0f, value));
    return static_cast<std::uint8_t>(value * 255.0f + 0.5f);
}

osg::Vec4ub get_color(osg::Array* array, std::size_t index) {
    if (array == nullptr || array->getNumElements() == 0) {
        return osg::Vec4ub(255, 255, 255, 255);
    }
    const std::size_t color_index =
        array->getNumElements() == 1
            ? 0
            : std::min(
                  index,
                  static_cast<std::size_t>(array->getNumElements() - 1)
              );
    if (auto* colors = dynamic_cast<osg::Vec4ubArray*>(array)) {
        return (*colors)[color_index];
    }
    if (auto* colors = dynamic_cast<osg::Vec3ubArray*>(array)) {
        const osg::Vec3ub color = (*colors)[color_index];
        return osg::Vec4ub(color.r(), color.g(), color.b(), 255);
    }
    if (auto* colors = dynamic_cast<osg::Vec4Array*>(array)) {
        const osg::Vec4 color = (*colors)[color_index];
        return osg::Vec4ub(
            float_color_to_byte(color.r()),
            float_color_to_byte(color.g()),
            float_color_to_byte(color.b()),
            float_color_to_byte(color.a())
        );
    }
    if (auto* colors = dynamic_cast<osg::Vec3Array*>(array)) {
        const osg::Vec3 color = (*colors)[color_index];
        return osg::Vec4ub(
            float_color_to_byte(color.x()),
            float_color_to_byte(color.y()),
            float_color_to_byte(color.z()),
            255
        );
    }
    return osg::Vec4ub(255, 255, 255, 255);
}

class GeometryPointVisitor : public osg::NodeVisitor {
public:
    GeometryPointVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}

    void apply(osg::Geode& geode) override {
        const osg::Matrixd transform = osg::computeLocalToWorld(getNodePath());
        for (unsigned int drawable_index = 0;
             drawable_index < geode.getNumDrawables();
             ++drawable_index) {
            osg::Geometry* geometry =
                geode.getDrawable(drawable_index)->asGeometry();
            if (geometry == nullptr || geometry->getVertexArray() == nullptr) {
                continue;
            }
            osg::Array* color_array = geometry->getColorArray();
            if (auto* vertices =
                    dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray())) {
                for (std::size_t index = 0; index < vertices->size(); ++index) {
                    append(osg::Vec3d((*vertices)[index]) * transform,
                           get_color(color_array, index));
                }
            } else if (auto* vertices =
                           dynamic_cast<osg::Vec3dArray*>(
                               geometry->getVertexArray())) {
                for (std::size_t index = 0; index < vertices->size(); ++index) {
                    append((*vertices)[index] * transform,
                           get_color(color_array, index));
                }
            }
        }
        traverse(geode);
    }

    void append(const osg::Vec3d& position, const osg::Vec4ub& color) {
        points.push_back(PointRecord{
            static_cast<float>(position.x()),
            static_cast<float>(position.y()),
            static_cast<float>(position.z()),
            color.r(),
            color.g(),
            color.b(),
            color.a(),
        });
    }

    std::vector<PointRecord> points;
};

void write_point_ply(const std::string& path,
                     const std::vector<PointRecord>& points) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create PLY file: " + path);
    }
    output << "ply\n"
           << "format binary_little_endian 1.0\n"
           << "comment CORE-3DVG geometry-only OSGB extraction\n"
           << "element vertex " << points.size() << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property uchar red\n"
           << "property uchar green\n"
           << "property uchar blue\n"
           << "property uchar alpha\n"
           << "end_header\n";
    for (const PointRecord& point : points) {
        output.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        output.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        output.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
        output.put(static_cast<char>(point.red));
        output.put(static_cast<char>(point.green));
        output.put(static_cast<char>(point.blue));
        output.put(static_cast<char>(point.alpha));
    }
    if (!output) {
        throw std::runtime_error("cannot write PLY file: " + path);
    }
}

int pack(const std::string& mesh_path,
         const std::string& payload_path,
         const std::string& output_path) {
    osg::ref_ptr<osg::Node> mesh = osgDB::readRefNodeFile(mesh_path);
    if (!mesh) {
        std::cerr << "Failed to read mesh: " << mesh_path << '\n';
        return 2;
    }

    const std::string payload = read_file(payload_path);
    const std::size_t chunk_count =
        (payload.size() + kChunkSize - 1) / kChunkSize;

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->setName("CORE-3DVG self-contained ScanNet scene");
    root->addChild(mesh);
    root->setUserValue(kSchemaKey, std::string("core3dvg-scannet-osgb-v1"));
    root->setUserValue(kEncodingKey, std::string("base64-zip"));
    root->setUserValue(kChunkCountKey, std::to_string(chunk_count));
    root->setUserValue(kPayloadSizeKey, std::to_string(payload.size()));

    for (std::size_t i = 0; i < chunk_count; ++i) {
        const std::size_t begin = i * kChunkSize;
        root->setUserValue(
            chunk_key(i),
            payload.substr(begin, std::min(kChunkSize, payload.size() - begin))
        );
    }

    if (!osgDB::writeNodeFile(*root, output_path)) {
        std::cerr << "Failed to write OSGB: " << output_path << '\n';
        return 3;
    }
    std::cout << "packed_payload_bytes=" << payload.size()
              << " chunks=" << chunk_count << '\n';
    return 0;
}

int convert_geometry(const std::string& mesh_path,
                     const std::string& output_path) {
    osg::ref_ptr<osg::Node> mesh = osgDB::readRefNodeFile(mesh_path);
    if (!mesh) {
        std::cerr << "Failed to read mesh: " << mesh_path << '\n';
        return 2;
    }
    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->setName("CORE-3DVG geometry-only scene");
    root->addChild(mesh);
    if (!osgDB::writeNodeFile(*root, output_path)) {
        std::cerr << "Failed to write OSGB: " << output_path << '\n';
        return 3;
    }
    return inspect(output_path);
}

int extract_geometry(const std::string& input_path,
                     const std::string& output_path) {
    osg::ref_ptr<osg::Node> root = osgDB::readRefNodeFile(input_path);
    if (!root) {
        std::cerr << "Failed to read OSGB: " << input_path << '\n';
        return 2;
    }
    GeometryPointVisitor visitor;
    root->accept(visitor);
    if (visitor.points.empty()) {
        std::cerr << "OSGB contains no extractable geometry vertices\n";
        return 8;
    }
    write_point_ply(output_path, visitor.points);
    std::cout << "extracted_geometry_vertices=" << visitor.points.size() << '\n';
    return 0;
}

int unpack(const std::string& input_path, const std::string& payload_path) {
    osg::ref_ptr<osg::Node> root = osgDB::readRefNodeFile(input_path);
    if (!root) {
        std::cerr << "Failed to read OSGB: " << input_path << '\n';
        return 2;
    }

    std::string schema;
    std::string encoding;
    std::string chunk_count_text;
    std::string payload_size_text;
    if (!root->getUserValue(kSchemaKey, schema)
        || !root->getUserValue(kEncodingKey, encoding)
        || !root->getUserValue(kChunkCountKey, chunk_count_text)
        || !root->getUserValue(kPayloadSizeKey, payload_size_text)) {
        std::cerr << "OSGB does not contain a CORE-3DVG bundle payload\n";
        return 4;
    }
    if (schema != "core3dvg-scannet-osgb-v1" || encoding != "base64-zip") {
        std::cerr << "Unsupported schema or encoding: "
                  << schema << " / " << encoding << '\n';
        return 5;
    }

    const std::size_t chunk_count = std::stoull(chunk_count_text);
    const std::size_t expected_size = std::stoull(payload_size_text);
    std::string payload;
    payload.reserve(expected_size);
    for (std::size_t i = 0; i < chunk_count; ++i) {
        std::string chunk;
        if (!root->getUserValue(chunk_key(i), chunk)) {
            std::cerr << "Missing payload chunk: " << i << '\n';
            return 6;
        }
        payload += chunk;
    }
    if (payload.size() != expected_size) {
        std::cerr << "Payload size mismatch: expected " << expected_size
                  << ", got " << payload.size() << '\n';
        return 7;
    }

    write_file(payload_path, payload);
    std::cout << "schema=" << schema
              << " payload_bytes=" << payload.size()
              << " chunks=" << chunk_count << '\n';
    return 0;
}

int inspect(const std::string& input_path) {
    osg::ref_ptr<osg::Node> root = osgDB::readRefNodeFile(input_path);
    if (!root) {
        std::cerr << "Failed to read OSGB: " << input_path << '\n';
        return 2;
    }

    GeometryStatsVisitor stats;
    root->accept(stats);
    std::string schema;
    std::string payload_size;
    root->getUserValue(kSchemaKey, schema);
    root->getUserValue(kPayloadSizeKey, payload_size);
    std::cout << "geometry_count=" << stats.geometry_count
              << " vertex_count=" << stats.vertex_count
              << " primitive_set_count=" << stats.primitive_set_count
              << " schema=" << (schema.empty() ? "none" : schema)
              << " payload_bytes="
              << (payload_size.empty() ? "0" : payload_size)
              << '\n';
    return 0;
}

void usage(const char* executable) {
    std::cerr
        << "Usage:\n"
        << "  " << executable
        << " pack <mesh.ply> <base64_payload.txt> <scene.osgb>\n"
        << "  " << executable
        << " convert-geometry <mesh.ply> <scene.osgb>\n"
        << "  " << executable
        << " unpack <scene.osgb> <base64_payload.txt>\n"
        << "  " << executable
        << " extract-geometry <scene.osgb> <points.ply>\n"
        << "  " << executable << " inspect <scene.osgb>\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "pack") {
            return pack(argv[2], argv[3], argv[4]);
        }
        if (argc == 4 && std::string(argv[1]) == "convert-geometry") {
            return convert_geometry(argv[2], argv[3]);
        }
        if (argc == 4 && std::string(argv[1]) == "unpack") {
            return unpack(argv[2], argv[3]);
        }
        if (argc == 4 && std::string(argv[1]) == "extract-geometry") {
            return extract_geometry(argv[2], argv[3]);
        }
        if (argc == 3 && std::string(argv[1]) == "inspect") {
            return inspect(argv[2]);
        }
        usage(argv[0]);
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 10;
    }
}
