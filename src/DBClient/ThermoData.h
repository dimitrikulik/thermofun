#ifndef THERMODATA_H
#define THERMODATA_H

// C++ includes
#include <memory>

// ThermoFun includes
#include "formuladata.h"

namespace ThermoFun {


class ThermoDataAbstract
{
public:

    ThermoDataAbstract( const string &name, const string &query, const vector<string> &paths, const vector<string> &headers, const vector<string> &names );

    /// Construct a copy of an ThermoDataAbstract instance
    ThermoDataAbstract(const ThermoDataAbstract& other);

    /// Destroy this instance
    virtual ~ThermoDataAbstract();

    // load values function
    /// Extract data connected to ReactionSet
//    virtual bsonio::ValuesTable  loadRecordsValues( const string& idReactionSet ) = 0;
    /// Extract data by condition
    virtual bsonio::ValuesTable  loadRecordsValues( const string& query, int sourcetdb,
                                                    const vector<ElementKey>& elements = {} ) = 0;
    /// Get Elements list from record
    virtual set<ElementKey> getElementsList( const string& idrec ) = 0;

    auto getDB() const -> boost::shared_ptr<bsonio::TDBGraph>;
    auto getName() const -> string;
    auto getQuery() const -> string;
    auto getDataNames() const -> vector<string>;
    auto getDataHeaders() const -> vector<string>;
    auto getDataFieldPaths() const -> vector<string>;
    auto getDataName_DataIndex() const -> std::map<std::string, int>;
    auto getDataName_DataFieldPath() const -> std::map<std::string, std::string>;
    auto getSubstSymbol_DefinesLevel() const -> std::map<std::string, std::string>;

    auto setDB(const boost::shared_ptr<bsonio::TDBGraph> &value) -> void;
    auto setDataNames(const vector<string> &value) -> void;
    auto setDataHeaders(const vector<string> &value) -> void;
    auto setDataFieldPaths(const vector<string> &value) -> void;
    auto setSubstSymbol_DefinesLevel(const std::map<std::string, std::string> &value) -> void;

    /**
     * @brief queryRecord returns a record queried by id
     * @param idRecord record id in the database
     * @param queryFields schema field paths which are copied and returned in the query result
     * @return string representing the result in JSON format containing the queryFields as keys
     */
    auto queryRecord(string idRecord, vector<string> queryFields) -> string;

protected:

    /// Test all elements from formula exist into list
    static auto testElementsFormula( const string& aformula, const vector<ElementKey>& elements) -> bool;

    /// Build ids list connected to idInVertex by edge
    auto getOutVertexIds(const string& edgeLabel, const string& idInVertex) -> vector<string>;

    /// Build ids list connected to idInVertex by edge, record edgesIds
    auto getOutVertexIds(const string& edgeLabel, const string& idInVertex,  vector<string> &edgesIds) -> vector<string>;

    auto resetDataPathIndex() -> void;

    auto setDefaultLevelForReactionDefinedSubst(bsonio::ValuesTable valuesTable) -> void;

    auto getDB_fullAccessMode() const -> boost::shared_ptr<bsonio::TDBGraph>;

    auto queryInEdgesDefines_(string idSubst, vector<string> queryFields,  string level) -> vector<string>;

    auto definesReactionSymbol_(string idSubst, string level) -> std::string;

    auto queryInEdgesTakes_(string idReact, vector<string> queryFields) -> vector<string>;

    auto reactantsCoeff_(string idReact) -> std::map<string, double>;

private:
    struct Impl;

    std::shared_ptr<Impl> pimpl;

};


}

#endif // THERMODATA_H
