#include <crow.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// PostgreSQL Connection Details
const std::string DB_CONN = "dbname=qdap_test user=amt password=amt host=192.168.2.23 port=5432";

// Function to Fetch Data from PostgreSQL
json queryDatabase(const std::string& sql) {
    json result = json::array();
    try {
        pqxx::connection conn(DB_CONN);
        pqxx::work txn(conn);
        pqxx::result res = txn.exec(sql);

        for (auto row : res) {
            json rowData;
            for (auto field : row) {
                rowData[field.name()] = field.as<std::string>();
            }
            result.push_back(rowData);
        }
        txn.commit();
    } catch (const std::exception& e) {
        result = {{"error", e.what()}};
    }
    return result;
}

// API Endpoint to Fetch Available Symbols
json getSymbols(const std::string& query) {
    std::string sql = "SELECT DISTINCT symbol FROM ohlcv_future_per_minute WHERE symbol ILIKE '%" + query + "%' LIMIT 10";
    return queryDatabase(sql);
}

// API Endpoint to Fetch Available Expiry Dates
json getExpiries(const std::string& symbol) {
    std::string sql = "SELECT DISTINCT expiry FROM ohlcv_future_per_minute WHERE symbol = '" + symbol + "' ORDER BY expiry";
    return queryDatabase(sql);
}

// API Endpoint to Fetch Available Strike Prices
json getStrikes(const std::string& symbol, const std::string& expiry, const std::string& type) {
    std::string sql = "SELECT DISTINCT strike FROM ohlcv_future_per_minute "
                      "WHERE symbol = '" + symbol + "' AND expiry = '" + expiry + "' AND opt_type = '" + type + "' "
                      "ORDER BY strike";
    return queryDatabase(sql);
}

int main() {
    crow::SimpleApp app;

    // Endpoint to Fetch Symbols
    CROW_ROUTE(app, "/api/get_symbols").methods(crow::HTTPMethod::GET)
    ([](const crow::request& req) {
        auto query = req.url_params.get("query");
        if (!query) return crow::response(400, "Missing query parameter");

        json data = getSymbols(query);
        return crow::response(data.dump());
    });

    // Endpoint to Fetch Expiry Dates
    CROW_ROUTE(app, "/api/get_expiries").methods(crow::HTTPMethod::GET)
    ([](const crow::request& req) {
        auto symbol = req.url_params.get("symbol");
        if (!symbol) return crow::response(400, "Missing symbol parameter");

        json data = getExpiries(symbol);
        return crow::response(data.dump());
    });

    // Endpoint to Fetch Strike Prices
    CROW_ROUTE(app, "/api/get_strikes").methods(crow::HTTPMethod::GET)
    ([](const crow::request& req) {
        auto symbol = req.url_params.get("symbol");
        auto expiry = req.url_params.get("expiry");
        auto type = req.url_params.get("type");

        if (!symbol || !expiry || !type) return crow::response(400, "Missing parameters");

        json data = getStrikes(symbol, expiry, type);
        return crow::response(data.dump());
    });

    // Start Server
    app.port(8080).multithreaded().run();
}

/**
 * 
 g++ -o backend backend.cpp -lcrow -lpqxx -lpq
./backend

 * / */