#include "TraversalData.h"

// ThermoFun includes
#include "jsonio/jsondomfree.h"
#include "SubstanceData.h"
#include "ReactionData.h"
#include "../Database.h"
#include "../Substance.h"
#include "../Reaction.h"
#include "../Common/ParseBsonTraversalData.h"
#include "../Common/Exception.h"
#include "../OptimizationUtils.h"

#include <sys/time.h>

struct timeval start, end_;

using namespace jsonio;

namespace ThermoFun
{

std::vector<std::string> queryFieldsVertex       = {"_id", "_label", "_type", "properties.symbol"};
std::vector<std::string> queryFieldsEdgeDefines  = { "_outV", "_label"};
std::vector<std::string> queryFieldsEdgeTakes    = { "_outV", "_label", "properties.stoi_coeff" };

enum DefinesLevelMode {
    all= 0,
    single,
    multiple
};

struct TraversalData::Impl
{
    /// map of substance symbol and level for the edge defines connected to the substance
    std::map<std::string, std::string> substSymbol_definesLevel;

    /// current level for the traversal operation
    std::string level = "0";

    /// mode of using levels for traversal (all: collect all connected; single: collect all connected using one
    /// defines level for all substances; multiple: collect all connectec using different levels for different
    /// substances (from map substSymbol_definesLevel)
    DefinesLevelMode definesLevelMode = DefinesLevelMode::single;

    /// map of database ids and bson record of vertexes
//    std::map<std::string, bson> idBsonRec;

    /// pointer to substance data for access to substance records
    SubstanceData_  *substData;

    /// pointer to reaction data for access to reaction records
    ReactionData_   *reactData;

    Impl(SubstanceData_ *asubstData,  ReactionData_ *areactData ):
     substData( asubstData ), reactData( areactData )
    {  }
};

TraversalData::TraversalData(SubstanceData_ *substData,  ReactionData_ *reactData) :
    pimpl(new Impl(substData, reactData ))
{ }

auto TraversalData::operator=(TraversalData other) -> TraversalData&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

//TraversalData::TraversalData(const TraversalData& other)
//: pimpl(new Impl(*other.pimpl))
//{}

TraversalData::~TraversalData()
{ }

// Public
auto TraversalData::getMapOfConnectedIds(vector<int> selNdx, vector<string> idsList, string level_) -> VertexId_VertexType
{
    pimpl->level = level_;
    pimpl->definesLevelMode = DefinesLevelMode::single;

    return getResult(idsList, selNdx);
}

auto TraversalData::getMapOfConnectedIds( vector<string> idList, string level_) -> VertexId_VertexType
{
    pimpl->level = level_;
    pimpl->definesLevelMode = DefinesLevelMode::single;

    return getResult(idList);
}

auto TraversalData::getMapOfConnectedIds(vector<string> idList, std::map<std::string, std::string> substSymbol_definesLevel) -> VertexId_VertexType
{
    pimpl->definesLevelMode = DefinesLevelMode::multiple;

    if (!substSymbol_definesLevel.empty())
    {
        pimpl->substSymbol_definesLevel = substSymbol_definesLevel;
    } else
    {
        pimpl->definesLevelMode = DefinesLevelMode::single;
    }
    return getResult(idList);;
}

auto TraversalData::getDatabaseFromMapOfIds(VertexId_VertexType resultTraversal, string level_) -> Database
{
    pimpl->level = level_;
    pimpl->definesLevelMode = DefinesLevelMode::single;
    return getDatabase(resultTraversal);
}

auto TraversalData::getDatabaseFromMapOfIds(VertexId_VertexType resultTraversal, std::map<std::string, std::string> substSymbol_definesLevel) -> Database
{
    if (!substSymbol_definesLevel.empty())
    {
        pimpl->definesLevelMode = DefinesLevelMode::multiple;
        pimpl->substSymbol_definesLevel = substSymbol_definesLevel;
    } else
    {
        pimpl->definesLevelMode = DefinesLevelMode::single;
    }
    return getDatabase(resultTraversal);
}

VertexId_VertexType TraversalData::getMapOfConnectedIds( vector<string> idList )
{
    pimpl->definesLevelMode = DefinesLevelMode::all;

    return getResult(idList);
}

// Private
auto TraversalData::getResult(vector<string> idList, vector<int> selNdx) -> VertexId_VertexType
{
    VertexId_VertexType result, r;

    if (selNdx.size() == 0)
    {
        for (uint ii=0; ii<idList.size(); ii++)
        {
            r = linkedDataFromId(idList[ii]);
            result.insert(r.begin(), r.end());
        }
    } else
    {
        for( uint ii=0; ii<selNdx.size(); ii++ )
        {
            r = linkedDataFromId(idList[selNdx[ii]]);
            result.insert(r.begin(), r.end());
        }
    }

    return result;
}

auto TraversalData::level (std::string idSubst) -> std::string
{
    bson record;
    std::string level_;

    switch(pimpl->definesLevelMode)
    {
    case DefinesLevelMode::all         : level_ = "-1";  // follows all connected data
        break;
    case DefinesLevelMode::single      : level_ = pimpl->level;; // follows edges defines with level
        break;
    case DefinesLevelMode::multiple    : {
        std::string substSymb; std::string key = idSubst +":";
        record = pimpl->substData->getJsonBsonRecordVertex(key).second;
        bsonio::bson_to_key( record.data, "properties.symbol", substSymb );
        if (pimpl->substSymbol_definesLevel.find(substSymb) != pimpl->substSymbol_definesLevel.end()) // follows edges defines with specific leveles for substSymbols
            level_ = pimpl->substSymbol_definesLevel[substSymb];   // if the substance symbol is not found in the map, it uses the default level
        else
            level_ = pimpl->level;
    }
        break;
    }

    return level_;
}

auto TraversalData::linkedDataFromId(std::string id_) -> VertexId_VertexType
{
    string _idRecord, _label, _type, _symbol, level_;
    bson record;
    VertexId_VertexType result;

    // get recrod
    record = pimpl->substData->getJsonBsonRecordVertex(id_+":").second;

    // Extract data from fields
    bsonio::bson_to_key( record.data, "_id",    _idRecord);
    bsonio::bson_to_key( record.data, "_label", _label);
    bsonio::bson_to_key( record.data, "_type",  _type);
    bsonio::bson_to_key( record.data, "properties.symbol",  _symbol);

    // get level mode
    level_ = level(_idRecord);

    // if not in the database proceed
    if (!result.count(_idRecord))
    {
        if (_label == "substance")
        {
            result.insert(std::pair<std::string,std::string>(_idRecord, "substance"));
            followIncomingEdgeDefines(_idRecord, result, level_);
        }
        if (_label == "reaction")
        {
            result.insert(std::pair<std::string,std::string>(_idRecord, "reaction"));
            followIncomingEdgeTakes(_idRecord, result);
        }
    }

    return result;
}

void TraversalData::followIncomingEdgeDefines(std::string _idSubst, VertexId_VertexType &result, string level_)
{
    string _idReact = "";
    bson record, _resultDataReac;
    vector<string> _resultDataEdge;

    _resultDataEdge = pimpl->substData->queryInEdgesDefines(_idSubst, queryFieldsEdgeDefines, level_ );
    for(uint i = 0; i < _resultDataEdge.size(); i++)
    {
        jsonToBson(&record, _resultDataEdge[i]);
        bsonio::bson_to_key( record.data, "_outV", _idReact );
        _resultDataReac = pimpl->reactData->getJsonBsonRecordVertex(_idReact+":").second /*pimpl->reactData->queryRecord( _idReact, queryFieldsVertex)*/;

        // if not in the database
        if (!result.count(_idReact))
        {
            result.insert((std::pair<std::string,std::string>(_idReact, "reaction")));
            followIncomingEdgeTakes(_idReact, result);
        }
    }
}

void TraversalData::followIncomingEdgeTakes(std::string _idReact, VertexId_VertexType &result)
{
    string _idSubst = ""; string level_ = "0";
    bson record, _resultDataSubst;
    vector<string> _resultDataEdge;

    _resultDataEdge = pimpl->reactData->queryInEdgesTakes(_idReact, queryFieldsEdgeTakes);
    for(uint i = 0; i < _resultDataEdge.size(); i++)
    {
        jsonToBson(&record, _resultDataEdge[i]);
        bsonio::bson_to_key( record.data, "_outV", _idSubst );
        _resultDataSubst = pimpl->substData->getJsonBsonRecordVertex(_idSubst+":").second/*pimpl->substData->queryRecord(_idSubst, queryFieldsVertex)*/;

        // if not in the database
        if (!result.count(_idSubst))
        {
            level_ = level(_idSubst);
            result.insert((std::pair<std::string,std::string>(_idSubst, "substance")));
            followIncomingEdgeDefines(_idSubst, result, level_);
        }
    }
}

auto TraversalData::getDatabase(VertexId_VertexType resultTraversal) -> Database
{
    string _idSubst, _idReact, substSymb; string level_ = pimpl->level;
    bson record;
    Database tdb;
    auto result = resultTraversal;

    // The set of all aqueous species in the database
    SubstancesMap substances_map;
    // The set of all gaseous species in the database
    ReactionsMap reactions_map;

    // get substances and the reaction symbol if necessary
    for(auto iterator = result.begin(); iterator != result.end(); iterator++)
    {
        if (iterator->second == "substance")
        {
            _idSubst = iterator->first;
            record = pimpl->substData->getJsonBsonRecordVertex(_idSubst+":").second;
            bsonio::bson_to_key( record.data, "properties.symbol", substSymb );

            level_ = level(_idSubst);
            Substance substance = parseSubstance(record.data);

            // get reaction symbol which define substance with _idSubst
            string definesReactSymb = pimpl->substData->definesReactionSymbol(_idSubst, level_);
            if (definesReactSymb != "")
            {
                substance.setReactionSymbol(definesReactSymb);
                substance.setThermoCalculationType(ThermoFun::SubstanceThermoCalculationType::type::REACDC);
            }

            if ( substances_map.find(substance.symbol()) == substances_map.end() ) {
                substances_map[substance.symbol()] = substance;
            } else {
                errorSameSymbol("substance", substance.symbol(), __LINE__, __FILE__ );
            }
        } else
            if (iterator->second == "reaction")
            {
                _idReact = iterator->first;
                record = pimpl->reactData->getJsonBsonRecordVertex(_idReact+":").second;

                Reaction reaction = ThermoFun::parseReaction(record.data);

                // get reactants by following reaction incoming takes edge
                reaction.setReactants(pimpl->reactData->reactantsCoeff(_idReact));

                if ( reactions_map.find(reaction.symbol()) == reactions_map.end() ) {
                    reactions_map[reaction.symbol()] = reaction;
                } else {
                    errorSameSymbol("reaction", reaction.symbol(), __LINE__, __FILE__ );
                }
            }
    }
    tdb.addMapReactions(reactions_map);
    tdb.addMapSubstances(substances_map);
    return tdb;
}

auto TraversalData::linkedDataIds(const std::vector<std::string> aKeyList, std::vector<std::string> &substIds, std::vector<std::string> &reactIds) -> void
{
    reactIds.clear(); substIds.clear();
    for (auto idType : getMapOfConnectedIds(aKeyList))
    {
        if (idType.second == "reaction")
            reactIds.push_back((idType.first));
        if (idType.second == "substance")
            substIds.push_back((idType.first));
    }
}


}
