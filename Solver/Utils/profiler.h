#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <stack>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include "luisa/core/clock.h"

namespace lcs
{

struct ProfileNode
{
    std::string name;
    double elapsed_ms{0.0};
    std::vector<std::unique_ptr<ProfileNode>> children;
    ProfileNode* parent{nullptr};
    int depth{0};
    
    ProfileNode(const std::string& n, ProfileNode* p = nullptr) 
        : name(n), parent(p), depth(p ? p->depth + 1 : 0) {}
    
    ProfileNode* add_child(const std::string& child_name)
    {
        children.push_back(std::make_unique<ProfileNode>(child_name, this));
        return children.back().get();
    }
    
    // Convert to JSON-like string without external dependencies
    std::string to_json_string() const
    {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"name\": \"" << escape_json(name) << "\",\n";
        oss << "  \"elapsed_ms\": " << elapsed_ms << ",\n";
        oss << "  \"depth\": " << depth << ",\n";
        oss << "  \"children\": [";
        for (size_t i = 0; i < children.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << "\n" << indent_json(children[i]->to_json_string(), 2);
        }
        if (!children.empty()) oss << "\n  ";
        oss << "]\n}";
        return oss.str();
    }
    
private:
    static std::string escape_json(const std::string& s)
    {
        std::ostringstream oss;
        for (char c : s)
        {
            switch (c)
            {
                case '"': oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\b': oss << "\\b"; break;
                case '\f': oss << "\\f"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default: oss << c;
            }
        }
        return oss.str();
    }
    
    static std::string indent_json(const std::string& json, int spaces)
    {
        std::ostringstream oss;
        std::istringstream iss(json);
        std::string line;
        std::string indent(spaces, ' ');
        while (std::getline(iss, line))
        {
            oss << indent << line << "\n";
        }
        std::string result = oss.str();
        if (!result.empty() && result.back() == '\n')
            result.pop_back();
        return result;
    }
};

class Profiler
{
public:
    static Profiler& instance()
    {
        static Profiler inst;
        return inst;
    }
    
    void reset()
    {
        root_ = std::make_unique<ProfileNode>("root");
        current_node_ = root_.get();
        while (!node_stack_.empty()) node_stack_.pop();
        node_stack_.push(root_.get());
    }
    
    void begin_frame(const std::string& frame_name)
    {
        reset();
        root_->name = frame_name;
    }
    
    void push_node(const std::string& name)
    {
        if (!current_node_) return;
        current_node_ = current_node_->add_child(name);
        node_stack_.push(current_node_);
        cpu_start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void pop_node()
    {
        if (node_stack_.size() <= 1) return;
        
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_elapsed = std::chrono::duration<double, std::milli>(cpu_end - cpu_start_time_).count();
        
        current_node_->elapsed_ms = cpu_elapsed;
        
        node_stack_.pop();
        if (!node_stack_.empty())
        {
            current_node_ = node_stack_.top();
        }
        cpu_start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    std::string get_result_json() const
    {
        if (!root_) return "{}";
        return root_->to_json_string();
    }
    
    void save_to_file(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (file.is_open())
        {
            file << get_result_json();
            file.close();
        }
    }
    
    void print_tree() const
    {
        if (!root_) return;
        print_node(root_.get(), 0);
    }
    
private:
    void print_node(const ProfileNode* node, int indent) const
    {
        std::string prefix(indent * 2, ' ');
        std::cout << prefix << node->name << ": " << node->elapsed_ms << " ms" << std::endl;
        for (const auto& child : node->children)
        {
            print_node(child.get(), indent + 1);
        }
    }
    
    std::unique_ptr<ProfileNode> root_;
    ProfileNode* current_node_{nullptr};
    std::stack<ProfileNode*> node_stack_;
    std::chrono::high_resolution_clock::time_point cpu_start_time_;
};

// Helper macros for easy profiling
#define PROFILE_FRAME(name) lcs::Profiler::instance().begin_frame(name)
#define PROFILE_PUSH(name) lcs::Profiler::instance().push_node(name)
#define PROFILE_POP() lcs::Profiler::instance().pop_node()

// RAII-based profile scope
class ProfileScope
{
public:
    explicit ProfileScope(const std::string& name)
    {
        Profiler::instance().push_node(name);
    }
    
    ~ProfileScope()
    {
        Profiler::instance().pop_node();
    }
};

#define PROFILE_SCOPE(name) lcs::ProfileScope _profile_scope_##__LINE__(name)

} // namespace lcs
